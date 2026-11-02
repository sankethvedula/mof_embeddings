#!/usr/bin/env python
"""Neighbours from global structure vectors vs neighbours from the atom-wise
structure FGW distance, for MOF queries whose FGW neighbours were judged
chemically meaningful (Q6 Mn, Q3 Co, Q4 Cd, Q1 Cu by default).

Global vectors pool the same z-scored UMA atom features that FGW uses (sr_1m
statistics), over every OMat24 sr_10m structure:
  zmean   Z-weighted mean, sum_i Z_i x_i / sum_i Z_i: the archive's H / sum Z after
          per-channel standardisation (H = sum_i Z_i h_i is the stored structure vector)
  mean    unweighted mean over atoms
Global distance = Euclidean distance between pooled vectors. Frames with atoms
closer than 0.8 A (flagged by the prefilter) are excluded, as in the search.

Stages
  pool     (Slurm job) pooled vectors for all 10M frames and their distances to
           every query -> $SCRATCH/global_pool/{zmean,mean}.npy, dist_{zmean,mean}.npy
  compare  for the chosen queries:
           - rank agreement between global distance and FGW over each query's
             50,000 reranked candidates (Spearman), and overlap of the top-k sets
           - global rank, over all 10M structures, of FGW's top-10 and of the
             neighbours rated "sensible" in mof_neighbours_per_subset.csv
           - the global method's own neighbours (top-10 distinct formulas overall,
             top-5 distinct per family in AIMD 1000 K, AIMD 3000 K and rattled relax),
             scored with exact FGW and the same verdict rules as the FGW neighbours
           -> data/mof_search/global_vs_fgw/{pairs.csv, summary.csv, verdicts.csv}
  plot     global distance vs FGW over the candidates, and verdict counts per method
           -> data/mof_search/global_vs_fgw/{scatter.png, verdicts.png}
"""
import argparse
import glob
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from mof_embeddings.config import ARCHIVE, SCRATCH
from mof_embeddings.mof_search.common import OUT, SAMPLE, SR10M, SR10M_LMDB, query_features, uma_stats
from mof_embeddings.mof_search.compare_subset_terms import FAMILY, ch_bonds
from mof_embeddings.mof_search.mof_neighbours_per_subset import add_verdicts, pair_facts
from mof_embeddings.structure_distance.structure_fgw import periodic_kernel_fast

POOL = Path(f"{SCRATCH}/global_pool")
RES = OUT / "global_vs_fgw"
L = f"{ARCHIVE}/derived/uma_atom_pca_sr_10m/frame_labels/row_labels"
VARIANTS = ["zmean", "mean"]
FAMILIES = ["AIMD 1000 K", "AIMD 3000 K", "rattled relax"]
KS = [5, 10, 20, 50]


# ---------- pooling ----------
def pooled(x, z, offsets):
    """Z-weighted and plain means of rows x split at offsets (relative to x)."""
    n = np.diff(offsets).astype(np.float64)
    zw = np.add.reduceat(x * z[:, None], offsets[:-1], axis=0) / np.add.reduceat(z, offsets[:-1])[:, None]
    mean = np.add.reduceat(x, offsets[:-1], axis=0) / n[:, None]
    return zw, mean


def query_vectors(mu, sd):
    q, feats, _ = query_features(mu, sd)
    emb = np.load(SAMPLE / "embeddings.npz")
    off = emb["frame_offsets"]
    zs = [emb["atomic_numbers"][off[i]:off[i + 1]].astype(np.float64) for i in q.idx]
    zw = np.stack([(f * z[:, None]).sum(0) / z.sum() for f, z in zip(feats, zs)])
    mean = np.stack([f.mean(0) for f in feats])
    return {"zmean": zw, "mean": mean}


def _pool_chunk(job):
    lo, hi, mu, sd, qv = job
    off = np.load(SR10M / "frame_atom_offsets.npy", mmap_mode="r")
    a, b = int(off[lo]), int(off[hi])
    x = (np.asarray(np.load(SR10M / "atom_latents.npy", mmap_mode="r")[a:b], dtype=np.float64) - mu) / sd
    z = np.asarray(np.load(SR10M / "atom_atomic_numbers.npy", mmap_mode="r")[a:b], dtype=np.float64)
    zw, mean = pooled(x, z, np.asarray(off[lo:hi + 1], dtype=np.int64) - a)
    for name, v in [("zmean", zw), ("mean", mean)]:
        np.load(POOL / f"{name}.npy", mmap_mode="r+")[lo:hi] = v.astype(np.float32)
        d = np.sqrt(((v[:, None, :] - qv[name][None]) ** 2).sum(-1))
        np.load(POOL / f"dist_{name}.npy", mmap_mode="r+")[lo:hi] = d.astype(np.float32)
    return hi - lo


def pool(procs, chunks):
    POOL.mkdir(parents=True, exist_ok=True)
    mu, sd = uma_stats()
    qv = query_vectors(mu, sd)
    n = len(np.load(SR10M / "frame_atom_offsets.npy", mmap_mode="r")) - 1
    for name in VARIANTS:
        np.lib.format.open_memmap(POOL / f"{name}.npy", "w+", np.float32, (n, len(mu)))
        np.lib.format.open_memmap(POOL / f"dist_{name}.npy", "w+", np.float32, (n, len(qv[name])))
    edges = np.linspace(0, n, chunks + 1).astype(int)
    jobs = [(lo, hi, mu, sd, qv) for lo, hi in zip(edges[:-1], edges[1:])]
    done = 0
    with Pool(procs) as p:
        for k in p.imap_unordered(_pool_chunk, jobs):
            done += k
            print(f"pooled {done:,}/{n:,} frames", flush=True)
    (POOL / "meta.json").write_text(json.dumps(dict(
        frames=int(n), variants=VARIANTS, queries=list(pd.read_csv(OUT / "queries.csv")["query"]),
        features="UMA-128 atom features z-scored with sr_1m statistics",
        zmean="sum_i Z_i x_i / sum_i Z_i", mean="mean_i x_i", distance="Euclidean"), indent=2))


# ---------- comparison ----------
def flagged():
    parts = [np.load(p) for p in sorted(glob.glob(str(OUT / "prefilter_parts" / "part_*.npz")))]
    fr = np.concatenate([p["frame"] for p in parts]); md = np.concatenate([p["min_dist"] for p in parts])
    return fr[md < 0.8]


def ranks_of(d, frames):
    """1-based rank of each frame under distances d (ties broken by order)."""
    return np.array([int((d < d[f]).sum()) + 1 for f in frames])


def compare(queries):
    from ase.db import connect
    from scipy.stats import spearmanr
    RES.mkdir(parents=True, exist_ok=True)
    const = json.loads((OUT / "constants.json").read_text())
    qtab = pd.read_csv(OUT / "queries.csv")
    names = json.load(open(L + ".json"))["names"]
    label = np.load(L + ".npy", mmap_mode="r")
    family_of = np.array([FAMILY[x] for x in names])
    fam_code = np.array([FAMILIES.index(FAMILY[x]) if FAMILY[x] in FAMILIES else -1 for x in names])[np.asarray(label)]
    rr = pd.read_parquet(OUT / "rerank_results.parquet")
    sens = pd.read_csv(OUT / "mof_neighbours_per_subset.csv")
    bad = flagged()
    dist = {v: np.load(POOL / f"dist_{v}.npy", mmap_mode="r") for v in VARIANTS}
    db = connect(SR10M_LMDB, type="aselmdb", readonly=True)
    mu, sd = uma_stats()
    _, qf, qa = query_features(mu, sd)
    off = np.load(SR10M / "frame_atom_offsets.npy", mmap_mode="r")
    feats = np.load(SR10M / "atom_latents.npy", mmap_mode="r")
    formula_cache = {}

    def formula(f):
        if f not in formula_cache:
            formula_cache[f] = db.get_atoms(id=int(f) + 1).get_chemical_formula(mode="hill", empirical=True)
        return formula_cache[f]

    def kernel(at):
        C, _ = periodic_kernel_fast(np.array(at.cell), at.positions, const["sigma"])
        return len(C) * C / const["k0"]

    def distinct(order, n, fam=None):
        out, seen = [], set()
        for f in order:
            f = int(f)
            if fam is not None and family_of[label[f]] != fam:
                continue
            fm = formula(f)
            if fm not in seen:
                seen.add(fm); out.append(f)
            if len(out) == n:
                break
        return out

    summary, pairs = [], []
    for qname in queries:
        qi = int(np.flatnonzero(qtab["query"] == qname)[0]); qrow = qtab.iloc[qi]
        mof = qa[qi]; Kq = kernel(mof); b1 = ch_bonds(mof)
        cand = rr[rr["query"] == qname].sort_values("fgw")
        fgw_sorted = cand.fgw.values
        best_any = fgw_sorted[0]
        dq = {v: np.array(dist[v][:, qi], dtype=np.float64) for v in VARIANTS}
        for v in VARIANTS:
            dq[v][bad] = np.inf
        row = dict(query=qname, mof_formula=qrow.formula, metal=qrow.metal, mof_atoms=int(qrow.natoms),
                   n_candidates=len(cand))
        for v in VARIANTS:
            gc = dq[v][cand.frame.values]
            row[f"spearman_{v}_vs_fgw_candidates"] = round(float(spearmanr(gc, fgw_sorted).statistic), 3)
            g_order = np.argsort(dq[v])[:5000]
            for k in KS:
                row[f"top{k}_overlap_{v}"] = len(set(g_order[:k]) & set(cand.frame.values[:k])) / k
            r_fgw10 = ranks_of(dq[v], cand.frame.values[:10])
            row[f"median_{v}_rank_of_fgw_top10"] = int(np.median(r_fgw10))
            row[f"best_{v}_rank_of_fgw_top10"] = int(r_fgw10.min())
            row[f"{v}_top10_in_candidates"] = int(np.isin(g_order[:10], cand.frame.values).sum())
        summary.append(row)

        def record(method, scope, rank, f, facts=None):
            at = db.get_atoms(id=int(f) + 1)
            if facts is None:
                x = (np.asarray(feats[off[f]:off[f + 1]], dtype=np.float64) - mu) / sd
                facts = pair_facts(qrow, mof, qf[qi], Kq, b1, at, x, kernel(at), const)
            r = dict(query=qname, mof_formula=qrow.formula, mof_metal=qrow.metal, mof_atoms=len(mof),
                     method=method, scope=scope, rank=rank, sr10m_frame=int(f), omat_subset=names[label[f]],
                     family=family_of[label[f]], match_formula=formula(f), match_atoms=len(at),
                     vs_best_for_this_mof=round(facts["fgw"] / best_any, 3),
                     fgw_rank_equivalent=int(np.searchsorted(fgw_sorted, facts["fgw"])) + 1, **facts)
            for v in VARIANTS:
                r[f"global_distance_{v}"] = round(float(dq[v][f]), 4)
                r[f"global_rank_{v}"] = int((dq[v] < dq[v][f]).sum()) + 1
            pairs.append(r)

        # FGW neighbours: top-10 distinct overall, and the per-family neighbours rated sensible before
        for rank, f in enumerate(distinct(cand.frame.values, 10), 1):
            record("FGW", "overall", rank, f)
        for _, s in sens[(sens["query"] == qname) & (sens.verdict == "sensible")].iterrows():
            record("FGW", s.subset_family, int(s.rank_in_subset), int(s.sr10m_frame))
        # global neighbours, scored the same way
        for v in VARIANTS:
            order = np.argsort(dq[v])[:20000]
            for rank, f in enumerate(distinct(order, 10), 1):
                record(f"global {v}", "overall", rank, f)
            for fam in FAMILIES:
                fam_order = np.argsort(np.where(fam_code == FAMILIES.index(fam), dq[v], np.inf))[:20000]
                for rank, f in enumerate(distinct(fam_order, 5, fam), 1):
                    record(f"global {v}", fam, rank, f)
        print(f"{qname} done", flush=True)

    df = add_verdicts(pd.DataFrame(pairs))
    df.to_csv(RES / "pairs.csv", index=False)
    pd.DataFrame(summary).to_csv(RES / "summary.csv", index=False)
    ver = (df.assign(scope_kind=np.where(df.scope == "overall", "top-10 overall", "per family"))
             .groupby(["query", "method", "scope_kind"]).verdict.value_counts().unstack(fill_value=0))
    ver.to_csv(RES / "verdicts.csv")
    print(pd.DataFrame(summary).T.to_string())
    print(ver.to_string())


# ---------- figures ----------
def plot(queries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
    qtab = pd.read_csv(OUT / "queries.csv")
    rr = pd.read_parquet(OUT / "rerank_results.parquet")
    pairs = pd.read_csv(RES / "pairs.csv")
    summ = pd.read_csv(RES / "summary.csv").set_index("query")
    d = np.load(POOL / "dist_zmean.npy", mmap_mode="r")
    fig, axes = plt.subplots(1, len(queries), figsize=(4.1 * len(queries), 3.9), facecolor=SURFACE, squeeze=False)
    for ax, q in zip(axes[0], queries):
        qi = int(np.flatnonzero(qtab["query"] == q)[0])
        c = rr[rr["query"] == q]
        g = np.asarray(d[c.frame.values, qi])
        ax.scatter(g, c.fgw, s=2, color="#c3c2b7", rasterized=True, label="reranked candidates")
        p = pairs[(pairs["query"] == q)]
        f = p[(p.method == "FGW") & (p.scope == "overall")]
        gz = p[(p.method == "global zmean") & (p.scope == "overall")]
        ax.scatter(f.global_distance_zmean, f.fgw, s=22, color="#2a78d6", label="FGW top-10", zorder=3)
        ax.scatter(gz.global_distance_zmean, gz.fgw, s=22, color="#eb6834", marker="D", label="global top-10", zorder=3)
        ax.set_title(f"{q} {qtab.metal[qi]} · {qtab.formula[qi]}\nSpearman {summ.loc[q, 'spearman_zmean_vs_fgw_candidates']:.2f}",
                     fontsize=10, color=INK)
        ax.set_xlabel("global distance (Z-weighted mean)", fontsize=9, color=INK2)
        ax.set_ylabel("structure FGW distance", fontsize=9, color=INK2)
        ax.tick_params(labelsize=8, colors=INK2); ax.set_facecolor(SURFACE)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        ax.grid(color=GRID, lw=0.6)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(RES / "scatter.png", dpi=150, facecolor=SURFACE); plt.close(fig)

    order = ["sensible", "partly", "not chemically meaningful"]
    colors = {"sensible": "#1baf7a", "partly": "#eda100", "not chemically meaningful": "#c3c2b7"}
    methods = ["FGW", "global zmean", "global mean"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE, sharey=True)
    for ax, kind, title in [(axes[0], "overall", "Top-10 distinct neighbours over 10M structures"),
                            (axes[1], "family", "Top-5 distinct per family (AIMD 1000 K, 3000 K, rattled relax)")]:
        sub = pairs[pairs.scope == "overall"] if kind == "overall" else pairs[pairs.scope != "overall"]
        if kind == "family":     # FGW rows per family hold only the sensible ones; use the full earlier table instead
            prev = pd.read_csv(OUT / "mof_neighbours_per_subset.csv")
            prev = prev[prev["query"].isin(queries)].assign(method="FGW")
            sub = pd.concat([prev[["query", "method", "verdict"]], sub[sub.method != "FGW"][["query", "method", "verdict"]]])
        labels, left = [], None
        tab = sub.groupby(["query", "method"]).verdict.value_counts(normalize=True).unstack(fill_value=0)
        rows = [(q, m) for q in queries for m in methods if (q, m) in tab.index]
        y = np.arange(len(rows))[::-1]
        left = np.zeros(len(rows))
        for v in order:
            w = np.array([tab.loc[r].get(v, 0) for r in rows])
            ax.barh(y, w, left=left, color=colors[v], edgecolor=SURFACE, linewidth=1.5, label=v)
            left += w
        ax.set_yticks(y, [f"{q} · {m}" for q, m in rows], fontsize=8.5, color=INK2)
        ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"], fontsize=8, color=INK2)
        ax.set_title(title, fontsize=10, color=INK)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        ax.set_facecolor(SURFACE)
    axes[1].legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1.0, -0.32), ncol=3)
    fig.tight_layout(); fig.savefig(RES / "verdicts.png", dpi=150, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)
    print("saved", RES / "scatter.png", RES / "verdicts.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["pool", "compare", "plot"], required=True)
    parser.add_argument("--queries", nargs="+", default=["Q6", "Q3", "Q4", "Q1"])
    parser.add_argument("--procs", type=int, default=16)
    parser.add_argument("--chunks", type=int, default=400)
    args = parser.parse_args()
    if args.stage == "pool":
        pool(args.procs, args.chunks)
    elif args.stage == "compare":
        compare(args.queries)
    else:
        plot(args.queries)


if __name__ == "__main__":
    main()
