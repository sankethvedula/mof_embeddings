# Report: atom embeddings, FGW neighbours and test-time adaptation for MOFs

Summary of every experiment run in this project (26 October – 12 November 2026). Methods, all numbers and
the module behind each result are in [ANALYSIS.md](ANALYSIS.md); how to rerun the fine-tuning
experiments, and the proposed next steps, are in [TTA_REPRODUCE.md](TTA_REPRODUCE.md). Module names
are relative to `mof_embeddings`.

## Contents

1. [Data](#1-data)
2. [Atom embeddings: what they encode](#2-atom-embeddings-what-they-encode)
3. [Matching atoms and structures](#3-matching-atoms-and-structures)
4. [MOF-off versus OMat24 and the 10M search](#4-mof-off-versus-omat24-and-the-10m-search)
5. [Test-time adaptation (TTA)](#5-test-time-adaptation-tta)
6. [Conclusions](#6-conclusions)
7. [Open items](#7-open-items)

---

## 1. Data

* Embedding archive `/scratch/gpfs/ROSENGROUP/shared/aryan/saved_embeddings` (1.9 TB): UMA-128
  (uma-s-1p2), OMAT0-128 (mace-omat-0-medium), MH1-96 and MH1-1024 (mace-mh-1) for 100.8M OMat24
  structures and 13.5M external structures (AM, MP-ALOE, MatPES, MAD, MOF-off). Per-atom banks exist
  for UMA and MH1-96 on OMat24 sr_1m / sr_10m and on 10% samples of the external datasets.
* Coordinates come from the source files in `/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/data`;
  bank row *i* = source frame *i* (checked for every dataset).
* 12,275 sr_10m frames have atoms closer than 0.8 Å and are excluded from all structure analyses.
* 468 archive files (40.6 GB) belong to another group and could not be read.

## 2. Atom embeddings: what they encode

| Finding | Evidence |
|---|---|
| MACE features are element labels; UMA encodes local environment | 10-NN element agreement: MH1-96 92%, UMA 63% (83% after per-channel standardisation); η²(element) 0.79 vs 0.26 |
| Every MACE interaction-layer invariant is element-pure | MH1 layers 1/2 and OMAT0 layer 2: 99–100% agreement, one 2D island per element; the readout keeps the most environment information |
| UMA forms one connected manifold | 18.7M-atom cuML UMAPs (same settings as Aryan's): MH1 islands (86% 2D agreement), UMA bands ordered by coordination (29%) |
| MOF-off atoms sit where OMat24 is sparse | joint 20.9M-atom maps: 93% (UMA) / 83% (MH1) of MOF-off atoms in pixels ≥ 10× enriched relative to OMat24; MP-ALOE, MatPES inside the OMat24 footprint |

Modules: `embeddings.atom_element_clustering`, `umap_atoms`, `extract_mace_atom_variants`,
`evaluate_mace_variants`, `joint_atom_umap` (ANALYSIS §2–4).

## 3. Matching atoms and structures

**Structures of different size.** Extensive sums of atom vectors fail (4% retrieval of a cell's own
supercell). H/ΣZ (UMA 64%) or optimal transport between atom sets (MH1 81%) work (§5).

**Atom correspondence between a cell and its supercell** (84 held-out materials, ground truth from
StructureMatcher). Embedding-only OT gets the element right (98–100%) but preserves only 51–52% of
neighbour pairs, the same as random assignment. Fused Gromov–Wasserstein (FGW) needs three changes
for crystals: tiling by an integer supercell matrix, a kernel summed over periodic images, and
100 restarts. With them it preserves 68–69% of neighbour pairs; the ceiling is 74%, and FGW beats OT
on 75–79 of 84 materials. The choice of embedding (UMA, MH1, any MACE layer) does not matter
significantly (§6).

**Size-intensive structure distance.** K = N · Σ_images exp(−d²/2σ²) with global scale constants and
σ = 3 Å, α = 0.5: 97% top-1 retrieval of a cell's own supercell (§7). This distance, on UMA atom
features, is the "structure FGW" used below.

## 4. MOF-off versus OMat24 and the 10M search

* MOF-off and OMat24 are chemically disjoint: 2 of 2,552 MOF-off compositions occur in OMat24, 99.4% of
  MOFs contain C and H against 0.11% of OMat24. In UMA space a third of MOF-off atoms (68% of C) lie
  outside OMat24's atom distribution. 98 of 100 MOFs are closer to another MOF than to any of 1,000
  random OMat24 structures (§8).
* **10M search** (10 MOF queries Q1–Q10, ten metals): an OT + neighbour-density surrogate prefilter
  (Spearman 0.90 with FGW) followed by exact FGW on the top 50,000 per query. About half of the queries
  get chemically consistent analogues (Co MOF → Mn coordination compound, Mn formate → Li Mn oxalate).
  Every best match is a near-equilibrium AIMD frame. Rattled 300/500/1000 frames are 25–30% farther
  because rattling stretches C–H bonds (62–64% of C–H bonds off by > 0.15 Å) (§9).
* **Global vectors versus FGW** (Q6, Q3, Q4, Q1). Z-weighted-mean UMA vectors find sensible
  neighbours for the small Mn formate. They miss the metal environment of mixed-ligand MOFs: for Q1
  (Cu), FGW's neighbours rank ~800,000th by global distance (§9).

## 5. Test-time adaptation (TTA)

**Question.** Does fine-tuning MACE-OMAT-0 on a few structures selected by FGW distance to a target
MOF reduce its errors on that MOF (forces, energy, stress)?

**Common protocol** (`tta.build_sets`, `tta.prepare`, `slurm/tta_finetune.slurm`, `slurm/tta_batch.slurm`).
* **Model and training:** Aryan's MACE fine-tuning settings (energy/force/stress loss 1/10/10, EMA,
  AMSGrad, float32, the foundation model's E0s, batch 4, seed 123), 60 epochs, all weights trained.
* **Data:** 20 training structures (ranks 1–20) and 5 validation structures (ranks 21–25). The target
  is never used for training or model selection.
* **Target set:** every MOF-off PBE frame of the target's framework (4–90 AIMD frames, 300–1000 K).
  It is scored before training and after epochs 1, 2, 3, 6, 11, 21, 41 and 60.
* **Metrics:** force MAE, offset-free energy MAE (mean |eᵢ − mean error|, which removes the constant
  level-of-theory offset), raw energy MAE and stress MAE. Results are given as % change against
  MACE-OMAT-0, with 95% bootstrap intervals over targets and Wilcoxon tests for paired comparisons.

**Level-of-theory mismatch.** OMat24 labels are PBE(+U for oxides/fluorides of Co, Cr, Fe, Mn, Mo,
Ni, V, W) without dispersion. MOF-off labels are PBE-D3(BJ) without U. Raw energies therefore carry
a 170–240 meV/atom offset that OMat24 data cannot remove, so forces and offset-free energies are the
measures that count.

### 5.1 Round 1: four queries, lr 1e-4

| Target | Before (eV/Å) | FGW | Global vector | Random |
|---|---|---|---|---|
| Q6 Mn | 0.223 | +11% | −1% | +12% |
| Q3 Co | 0.220 | −2% | −1% | +11% |
| Q4 Cd | 0.156 | −11% | −10% | +8% |
| Q1 Cu | 0.245 | −4% | +5% | +15% |

Random structures always hurt. Neighbours give modest gains on three of the four targets.

### 5.2 Round 2: ten queries, 100 structures, lower learning rates (120 runs)

Mean force change after 60 epochs: FGW +1.2% / −0.1% / −0.6% at lr 1e-4 / 3e-5 / 1e-5, global vector
+4.2% / +0.8% / −1.9%, random +9.6% / +3.7% / +2.2%. 100 structures do not help (FGW +1.7%, global
+5.6%, random +13.3% at 1e-4). Lower learning rates shrink every change toward zero. Averaged over
targets, neighbours are neutral but always better than random.

### 5.3 Which FGW term predicts the gain (`tta.neighbour_terms`)

Each training set's FGW is split into a feature term (UMA cost of matched atoms) and a geometry term.
On 30 targets (120 neighbour sets, lr 3e-5):
* **Across sets**, the feature term predicts the force change: Spearman ρ = +0.52 (+0.62 at target
  level, +0.36 without +U targets). The geometry term has ρ = 0.00.
* **Within a target**, no term picks the best set (|ρ| ≤ 0.14; the lowest-feature set wins 9/30 times,
  chance 7.5).

The feature term identifies targets that OMat24 can help; it cannot choose between sets for one
target.

### 5.4 Round 3: 30 targets, FGW α, seeds, stopping, training variants

20 further targets (`tta.targets`) were added to the ten queries. Each was fine-tuned on five
20-structure sets: FGW α 0.5 / 0.25 / 0.1 re-ranked within a candidate pool (`tta.select_alpha`),
the global vector and random. lr 3e-5, 60 epochs.

| Metric (60 epochs) | FGW α 0.5 | FGW α 0.25 | FGW α 0.1 | Global | Random |
|---|---|---|---|---|---|
| Force MAE | +1.4% | +0.6% | +0.3% | +1.7% | +5.3% |
| Energy, offset-free | −2.7% | −4.1% | −1.9% | −2.7% | +6.8% |
| Stress MAE | −10% | −16% | −13% | −11% | +39% |

* Neighbours beat random clearly (FGW α 0.1 vs random: force −5.0 points, p = 0.002) but do not beat
  the global vector.
* **+U targets:** on the 10 targets whose metal is a +U metal, every set degrades forces (+2.4 to +5.8%).
* **Seeds:** a second seed changes the errors by 0.1–0.3 points, so the differences between targets
  are not training noise.
* **Stopping:** the target's error is lowest after 1–6 epochs and drifts up afterwards. Validation on
  held-out neighbours does not find that epoch.
* **Training variants:** readouts only, products + readouts, LoRA rank 4, stress weight 1000. None
  changes forces by more than ~1 point against training all weights.

### 5.5 Round 4: FGW α 0.25 on 79 targets (`tta.effect --cohorts fgwmid`)

49 more targets (every remaining MOF-off sample structure with ≥ 20 PBE frames and ≤ 200 atoms; mostly
Cu and Zn) bring the total to 79.

| Targets | Force, 6 epochs | Force, 60 epochs | Offset-free energy, 6 epochs | Stress, 6 epochs |
|---|---|---|---|---|
| all 79 | **−0.9%** [−1.8, −0.0], 48/79, p = 0.003 | −0.5% (n.s.) | **−4.3%**, p = 0.003 | −13% |
| 57 without a +U metal | −1.3% | −1.6% | −4.0% | −14% |
| 22 with a +U metal | +0.2% | +2.2% | −5.1% | −10% |

With 79 targets the small gain becomes significant, but it remains about 1% on forces.

### 5.6 Adapt or not: a target-blind rule (`tta.abstain`)

Before fine-tuning, decide from information available without target labels whether to fine-tune;
otherwise keep MACE-OMAT-0 (change 0). Rules: skip +U-metal targets; fine-tune only if the training
set's mean FGW feature term is below τ. τ is chosen leave-one-target-out, so no target is scored with
a τ fitted on it. FGW α 0.25, 79 targets:

| Rule | Epochs | Targets fine-tuned | Force | Offset-free energy | Stress | Targets worse by > 1% |
|---|---|---|---|---|---|---|
| always | 6 | 79 | −0.92% | −4.31% | −12.8% | 13 |
| feature < τ (τ ≈ 0.83) | 6 | 62 | **−1.34%** | −2.95% | −12.2% | 5 |
| no +U metal | 6 | 57 | −0.97% | −2.88% | −9.9% | 5 |
| both | 6 | 51 | −1.08% | −2.41% | −10.1% | 4 |
| always | 60 | 79 | −0.55% | −1.55% | −14.4% | 30 |
| both | 60 | 45 | **−1.48%** | +0.37% | −9.0% | 12 |

Abstaining removes most of the targets that get worse; the average gain stays small.

### 5.7 Weight interpolation (`tta.interpolate`, `slurm/tta_interp.slurm`)

w(λ) = (1 − λ) w_foundation + λ w_fine-tuned, evaluated on the target with no retraining. λ = 0 and
λ = 1 reproduce the stored evaluations exactly (316 checks). FGW α 0.25, 79 targets, % change:

| Fine-tuned epochs | λ | Force | Offset-free energy | Stress | Raw energy | Improved / worse by > 1% |
|---|---|---|---|---|---|---|
| 6 | 1.0 (no interpolation) | −0.9% | −4.2% | −13.7% | −21% | 48 / 13 |
| 60 | 1.0 (no interpolation) | −0.6% | −1.6% | −14.4% | −27% | 43 / 30 |
| 60 | 0.25 | −1.1% | −3.1% | −5.9% | −7% | 60 / 6 |
| 60 | **0.5** | **−1.6%** [−2.4, −0.8], p < 10⁻⁴ | −4.5% | −10.5% | −14% | 55 / 14 |
| 60 | 0.75 | −1.5% | −4.0% | −13.5% | −21% | 50 / 25 |

* Fine-tuning for 60 epochs and then interpolating halfway beats the best fixed early stop (6 epochs,
  λ = 1) by 0.7 points (47/79, p = 0.011).
* On the 57 targets without a +U metal it gives −2.3% on forces (45/57). The +U targets stay at about
  0% for any λ.

### 5.8 Round 5: MOF-off neighbours and +U-free OMat24 neighbours

Two new training sources at FGW α 0.25, lr 3e-5:
* **fgwmidnou:** the OMat24 pool without +U-labelled structures. This changes the set only for the 21
  targets that have +U structures among their nearest neighbours. The Mn and Mo targets whose whole
  pool is +U get a second pool of 1,000 structures without +U (`select_alpha --no-plus-u`).
* **moffgw:** MOF-off PBE frames of *other* frameworks (`tta.select_mof`), the target's own framework
  excluded, one frame per framework. These labels are at the target's level of theory (PBE-D3(BJ),
  no U).

All 79 targets, lr 3e-5:

| Training set | Epochs | Force | Offset-free energy | Stress | Raw energy |
|---|---|---|---|---|---|
| OMat24 FGW α 0.25 (round 4, 79 targets) | 6 / 60 | −0.9% / −0.5% | −4.3% / −1.6% | −13% / −14% | −21% / −27% |
| OMat24 without +U labels | 6 / 60 | −0.9% / −0.2% | −1.6% / +3.6% | −9.5% / −4.9% | −20% / −25% |
| **MOF-off, other frameworks** | 6 | **−8.9%** [−10.7, −7.1], 73/79 | −7.3% | −24% | −52% |
| **MOF-off, other frameworks** | 60 | **−21.2%** [−24.2, −18.3], 78/79 | **+23%** | −16% | −84% |

* **MOF-off neighbours reduce force errors ten to twenty times more than OMat24 neighbours.** Paired
  against OMat24 FGW on the same targets the difference is −8.0 points at 6 epochs (71/79,
  p = 1 × 10⁻¹²) and −20.7 points at 60 epochs (73/79, p = 5 × 10⁻¹⁴). The gain holds for +U metals
  (−24% at 60 epochs, 22/22) and other metals (−20%), and for queries, round-3 and round-4 targets alike.
* **Long training harms energies.** After 60 epochs the offset-free energy error rises by 23%
  (+52% on +U targets; paired against OMat24 FGW +24 points, p = 0.008) while forces keep improving. At 6 epochs energy and forces improve together.
  Weight interpolation is the obvious next fix.
* **Removing +U labels from OMat24 does not help the +U targets** (+3.6% on forces at 60 epochs
  against +2.2% with +U). The OMat24 problem is distance in chemistry, not only the +U labels.
* **Caveat 1, near-duplicates.** Only the target's own framework ID is excluded. 25 of 79 targets
  have a neighbour with the same empirical formula (another CSD entry or polymorph). These gain more
  (−32.8% at 60 epochs, −15.6% at 6) than the others (−15.9%, −5.8%), and the others still gain far more than with
  OMat24.
* **Caveat 2, FGW versus level of theory.** The gain tracks FGW distance across targets (Spearman
  +0.4 to +0.5 with the best neighbour's FGW and with the set's feature term, on the first 62 targets). But part of it may come
  only from matching the target's level of theory.

**Controls queued** (183 runs, Slurm chain 17429425–17429430):
* **mofrandom:** 20 random MOF-off frameworks.
* **mofglobal:** the 20 MOF-off frameworks nearest by the global vector.
* **moffgwnf:** MOF-off FGW without frameworks of the target's empirical formula. It differs from
  moffgw on 25 targets and is identical on the other 54.

Together these separate "FGW selection" from "same level of theory" and from near-duplicates.

## 6. Conclusions

1. Geometry is needed for atom correspondence: FGW with a periodic kernel preserves local
   connectivity, while embeddings alone do not. The embedding model is secondary.
2. MACE features are element labels; UMA features carry the local environment, which is why the FGW
   search uses UMA.
3. MOF-off is chemically disjoint from OMat24. OMat24's closest analogues are near-equilibrium AIMD
   frames, and rattled frames distort C–H bonds.
4. Fine-tuning MACE-OMAT-0 on 20 OMat24 neighbours gives at most −1 to −2% on forces, −4% on
   offset-free energy and −13% on stress. FGW selection avoids the damage done by random data,
   interpolation (λ 0.5) and a target-blind abstain rule make the gain reliable, and +U-metal targets
   are not helped.
5. Fine-tuning on 20 MOF-off frames of *other* frameworks, selected by FGW, cuts the target's force
   error by about 9% after 6 epochs and about 21% after 60 epochs (79 targets), but long training raises
   the offset-free energy error by about 23%. Whether FGW selection is
   responsible, rather than the shared level of theory, is what the queued controls will show.

## 7. Open items

* Finish the MOF-off controls (running). Then compare moffgw with mofrandom and
  mofglobal, and moffgwnf with moffgw.
* Weight interpolation of the MOF-off runs, to recover the offset-free energy lost at 60 epochs.
* Replay or L2-to-foundation regularisation on the winning data source (the MACE fork supports replay
  via `multiheads_finetuning` / `pt_train_file`).
* Scale up and change the model: many more targets, and the same protocol on UMA. See
  [TTA_REPRODUCE.md § Next steps](TTA_REPRODUCE.md#next-steps-for-aryan).
