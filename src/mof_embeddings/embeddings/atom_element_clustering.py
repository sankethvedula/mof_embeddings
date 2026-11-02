"""Does element identity organise the atom embeddings?

Samples whole OMat24 sr_1m structures and, in several spaces (raw UMA-128,
UMA PCA, UMA UMAP, raw MH1-96), measures:
  - eta^2: fraction of total variance explained by the element label
  - kNN (k=10): element accuracy, and neighbour fractions that share the
    element / the parent structure
"""
import sys, json, numpy as np, torch

from mof_embeddings.config import ARCHIVE, RUNS

A = f"{ARCHIVE}/"
U = A + "embeddings/uma/atom/omat24_sr1m/"
M = A + "embeddings/mace_mh1/atom/omat24_sr1m/readout96/"
P = A + "derived/uma_atom_pca_sr_1m/"
UMAP = A + "derived/uma_omat24_umap/atomwise_sr1m_external10pct_seed42/umap_raw128.npy"
N_FRAMES, K, SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5000, 10, 0
torch.set_num_threads(16)

off = np.load(U + "frame_atom_offsets.npy")
rng = np.random.default_rng(SEED)
frames = np.sort(rng.choice(len(off) - 1, N_FRAMES, replace=False))
rows = np.concatenate([np.arange(off[f], off[f + 1]) for f in frames])
frame_of = np.repeat(np.arange(N_FRAMES), np.diff(off)[frames])
Z = np.asarray(np.load(U + "atom_atomic_numbers.npy", mmap_mode="r")[rows]).astype(int)
G = np.asarray(np.load(U + "labels_element_group/row_labels.npy", mmap_mode="r")[rows]).astype(int)
S = np.asarray(np.load(U + "labels_subdataset/row_labels.npy", mmap_mode="r")[rows]).astype(int)

def take(path, cols=None):
    a = np.load(path, mmap_mode="r")
    x = np.asarray(a[rows]) if cols is None else np.asarray(a[rows, :cols])
    return x.astype(np.float32)

uma = take(U + "atom_latents.npy")
spaces = {
    "UMA raw 128D": uma,
    "UMA raw 128D, z-scored": (uma - uma.mean(0)) / uma.std(0),
    "UMA PCA 25D": take(P + "pc25_all.npy"),
    "UMA PCA 2D": take(P + "pc25_all.npy", 2),
    "UMA UMAP 2D": take(UMAP),
    "MH1 raw 96D": take(M + "atom_latents.npy"),
}

def eta2(x, y):
    tot = ((x - x.mean(0)) ** 2).sum()
    within = sum(((x[y == e] - x[y == e].mean(0)) ** 2).sum() for e in np.unique(y))
    return 1 - within / tot

def knn(x, k=K, chunk=4096):
    x = torch.from_numpy(x)
    idx = []
    for i in range(0, len(x), chunk):
        d = torch.cdist(x[i:i + chunk], x)
        d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float("inf")
        idx.append(d.topk(k, largest=False).indices)
    return torch.cat(idx).numpy()

p = np.bincount(Z) / len(Z)
out = {"n_frames": N_FRAMES, "n_atoms": int(len(rows)), "n_elements": int((p > 0).sum()),
       "majority_baseline": float(p.max()), "random_pair_same_element": float((p ** 2).sum()),
       "spaces": {}}
print(f"{N_FRAMES} structures, {len(rows)} atoms, {out['n_elements']} elements; "
      f"majority-class acc {p.max():.3f}, random-neighbour same-element {(p**2).sum():.3f}")
for name, x in spaces.items():
    nb = knn(x)
    same_el = Z[nb] == Z[:, None]
    same_fr = frame_of[nb] == frame_of[:, None]
    vote = np.array([np.bincount(r).argmax() for r in Z[nb]])
    r = {"eta2_element": float(eta2(x.astype(np.float64), Z)),
         "eta2_element_group": float(eta2(x.astype(np.float64), G)),
         "eta2_subdataset": float(eta2(x.astype(np.float64), S)),
         "knn_element_acc": float((vote == Z).mean()),
         "nbr_same_element": float(same_el.mean()),
         "nbr_same_structure": float(same_fr.mean()),
         "nbr_same_element_other_structure": float((same_el & ~same_fr).mean()),
         "nbr_other_element_same_structure": float((~same_el & same_fr).mean())}
    out["spaces"][name] = r
    print(f"{name:24s} " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
(RUNS / "atom_element_clustering").mkdir(parents=True, exist_ok=True)
json.dump(out, open(RUNS / "atom_element_clustering" / f"atom_element_clustering_{N_FRAMES}.json", "w"), indent=2)
