"""Shared pieces for searching OMat24 sr_10m for the best FGW matches of MOF queries.

Features are UMA-128 atom embeddings z-scored with OMat24 sr_1m statistics (sr_10m
has no MH1 atom bank). The prefilter summarises each structure's atoms by the
per-channel mean and standard deviation; the 2-Wasserstein distance between two
such diagonal Gaussians is sqrt(|mu1 - mu2|^2 + |s1 - s2|^2), a cheap stand-in for
the atom-level OT/FGW cost.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from mof_embeddings.config import ARCHIVE, DATA, PRETRAIN
from mof_embeddings.matching.ot_atom_mapping import BANKS, bank_stats

A = Path(f"{ARCHIVE}/embeddings")
SR10M = A / "uma/atom/omat24_sr10m"
SR10M_LMDB = f"{PRETRAIN}/data/processed/omat24/stratified_random_nested/sr_10m/train.lmdb"
SAMPLE = Path(f"{DATA}/structure_fgw/sample")
OUT = Path(f"{DATA}/mof_search")


def uma_stats():
    return bank_stats(BANKS["UMA"])


def query_features(mu, sd):
    """z-scored UMA atom features and structures of the 10 MOF queries."""
    from ase.io import read
    q = pd.read_csv(OUT / "queries.csv")
    emb = np.load(SAMPLE / "embeddings.npz")
    off = emb["frame_offsets"]
    atoms = read(SAMPLE / "structures.extxyz", ":")
    feats = [(emb["atom_uma128"][off[i]:off[i + 1]] - mu) / sd for i in q.idx]
    return q, feats, [atoms[i] for i in q.idx]


def summary(x):
    return np.concatenate([x.mean(0), x.std(0)])


def segment_summaries(x, offsets):
    """Per-structure [mean, std] of rows x split at offsets (offsets relative to x)."""
    n = np.diff(offsets)
    s1 = np.add.reduceat(x, offsets[:-1], axis=0)
    s2 = np.add.reduceat(x * x, offsets[:-1], axis=0)
    mean = s1 / n[:, None]
    std = np.sqrt(np.maximum(s2 / n[:, None] - mean ** 2, 0))
    return np.hstack([mean, std])
