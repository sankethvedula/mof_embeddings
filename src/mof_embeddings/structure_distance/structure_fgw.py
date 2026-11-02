#!/usr/bin/env python
"""FGW distance between two arbitrary periodic structures (different materials,
different atom counts, no supercell relation).

Each structure is a set of atoms with uniform weights 1/N and
  features  UMA+MH1 atom embeddings (z-scored per channel with OMat24 sr_1m
            statistics, each block scaled by 1/sqrt(width), as in fgw_atom_mapping.py)
  geometry  an intensive periodic neighbour kernel
              K[a,b] = N * sum_L exp(-|r_b + L - r_a|^2 / 2 sigma^2) / k0
            The factor N makes K independent of the cell chosen: in a k-fold
            supercell each pair's images are split over k copies while N grows k-fold,
            so averages over copies equal the primitive cell's values. k0 is one global
            constant (median per-atom kernel sum over a reference set) that puts K on
            an O(1) scale for every structure.
FGW objective: (1 - alpha) <M, T> + alpha * sum (K1[a,b] - K2[x,y])^2 T[a,x] T[b,y],
with M the squared feature distance divided by a global constant (median over
random atom pairs). Global constants keep distances comparable across all pairs.
Solved from the plain-OT plan and the uniform plan; the lower objective is the distance.
"""

from mof_embeddings.config import use_local_packages
use_local_packages()
import numpy as np
import ot
from scipy.spatial.distance import cdist


def periodic_kernel(structure, sigma):
    """Sum over all periodic images within 4 sigma, own images included (no N factor)."""
    C = np.eye(len(structure))
    for a, neighbours in enumerate(structure.get_all_neighbors(4 * sigma)):
        for n in neighbours:
            C[a, n.index] += np.exp(-n.nn_distance ** 2 / (2 * sigma ** 2))
    return (C + C.T) / 2


def periodic_kernel_fast(cell, positions, sigma):
    """Same kernel as periodic_kernel, vectorised over lattice images (ase-style cell/positions).
    Also returns the shortest interatomic distance, to flag unphysical frames."""
    cutoff = 4 * sigma
    heights = 1 / np.linalg.norm(np.linalg.inv(cell).T, axis=1)          # spacing of lattice planes
    n = np.ceil(cutoff / heights).astype(int) + 1
    grid = np.stack(np.meshgrid(*[np.arange(-k, k + 1) for k in n], indexing="ij"), -1).reshape(-1, 3)
    frac = np.linalg.solve(cell.T, positions.T).T % 1.0
    pos = frac @ cell
    diff = pos[None, :, :] - pos[:, None, :]                              # (N, N, 3): r_b - r_a
    C = np.zeros((len(pos), len(pos)))
    min_d2 = np.inf
    for shifts in np.array_split(grid @ cell, max(1, len(grid) * len(pos) ** 2 // 2_000_000)):
        r2 = ((diff[:, :, None, :] + shifts[None, None]) ** 2).sum(-1)    # (N, N, S)
        C += np.where(r2 <= cutoff ** 2, np.exp(-r2 / (2 * sigma ** 2)), 0).sum(-1)
        min_d2 = min(min_d2, r2[r2 > 1e-8].min(initial=np.inf))
    return (C + C.T) / 2, float(np.sqrt(min_d2))


def intensive_kernel(structure, sigma, k0):
    return len(structure) * periodic_kernel(structure, sigma) / k0


def fgw_distance(f1, K1, f2, K2, alpha, m0, max_iter=200, return_plan=False):
    """FGW objective at the best of two starts; alpha = 0 gives the plain OT (Wasserstein) cost.
    With return_plan, also returns the transport plan (atoms of structure 1 x atoms of structure 2)."""
    M = cdist(f1, f2, "sqeuclidean") / m0
    p, q = ot.unif(len(f1)), ot.unif(len(f2))
    T0 = ot.emd(p, q, M)
    if alpha == 0:
        value, plan = float(np.sum(T0 * M)), T0
    else:
        value, plan = np.inf, None
        for G0 in [T0, None]:
            T, log = ot.gromov.fused_gromov_wasserstein(M, K1, K2, p, q, loss_fun="square_loss", alpha=alpha,
                                                        G0=G0, log=True, max_iter=max_iter)
            if log["fgw_dist"] < value:
                value, plan = float(log["fgw_dist"]), T
    return (value, plan) if return_plan else value
