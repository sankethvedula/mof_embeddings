# Analysis description

This document describes the analyses in this repository: their aims, the methods, the numbers
obtained, and the module that produces each result. The slide deck
([`slides/atom_matching_2026-10-26.pdf`](../slides/atom_matching_2026-10-26.pdf)) presents the
same results with one finding per slide. Module names below are relative to `mof_embeddings`.

## Contents

1. [Data](#1-data)
2. [What atom embeddings encode](#2-what-atom-embeddings-encode)
3. [MACE feature variants](#3-mace-feature-variants)
4. [Joint atom maps of five datasets](#4-joint-atom-maps-of-five-datasets)
5. [Comparing structures of different cell size](#5-comparing-structures-of-different-cell-size)
6. [Atom correspondence between a cell and its supercell](#6-atom-correspondence-between-a-cell-and-its-supercell)
7. [A size-intensive structure distance](#7-a-size-intensive-structure-distance)
8. [MOF-off versus OMat24](#8-mof-off-versus-omat24)
9. [Search of 10M OMat24 structures for MOF analogues](#9-search-of-10m-omat24-structures-for-mof-analogues)
10. [Conclusions, limitations and next steps](#10-conclusions-limitations-and-next-steps)
11. [Where each result comes from](#11-where-each-result-comes-from)
12. [Test-time adaptation by fine-tuning on FGW neighbours](#12-test-time-adaptation-by-fine-tuning-on-fgw-neighbours)

---

## 1. Data

### Embedding archive

The shared archive (`/scratch/gpfs/ROSENGROUP/shared/aryan/saved_embeddings`, 1.9 TB, 49,586 files)
contains four representations computed for 100.8M OMat24 structures and 13.5M structures from
external datasets.

| Representation | Model | Layer | Width | Per-atom bank |
|---|---|---|---|---|
| UMA-128 | uma-s-1p2 | input to the energy block | 128 | yes |
| OMAT0-128 | mace-omat-0-medium | last-layer invariant node features | 128 | no |
| MH1-96 | mace-mh-1 | readout hidden units of all six heads | 96 | yes |
| MH1-1024 | mace-mh-1 | invariants of both interaction layers | 1,024 | no |

The MH1 readout spans six heads (`matpes_r2scan`, `mp_pbe_refit_add`, `spice_wB97M`,
`oc20_usemppbe`, `omol`, `omat_pbe`), each owning a contiguous block of 16 channels; `omat_pbe` is
channels 80–95.

Per-structure vectors are Z-weighted sums of atom vectors, H = Σᵢ Zᵢ hᵢ, and are therefore
extensive: a 2× supercell has twice the vector of its primitive cell. Per-atom banks exist for
UMA-128 and MH1-96 on the OMat24 sr_1m (1M structures, 18.7M atoms) and sr_10m samples, and on 10%
samples of the external datasets.

| Dataset | Structures | Per-atom sample |
|---|---|---|
| OMat24 | 100.8M | 1M (sr_1m) and 10M (sr_10m) |
| AM | 12.0M | 10% (UMA only) |
| MP-ALOE | 910k | 10% |
| MatPES (r²SCAN) | 388k | 10% |
| MAD | 96k | 10% |
| MOF-off (r²SCAN) | 80.6k | 10% (8,064 frameworks) |

### Coordinates and access

The archive stores atomic numbers but no coordinates. Structures are read from the source files in
`/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/data` (OMat24 aselmdb shards, the sr_1m and sr_10m
LMDBs, and `.traj` files for the external datasets). Bank row *i* corresponds to source frame *i*
(LMDB id *i* + 1); this was verified for every dataset by comparing atomic numbers. LMDBs must be
opened read-only. 468 files (40.6 GB) are restricted to another group; the UMA banks affected (AM,
MOF-off) were recovered from a combined external bank.

### Exclusions

12,275 sr_10m frames contain atoms closer than 0.8 Å (for example MoNaO₆P with a 0.65 Å P–O
distance). They produce extreme embeddings and are excluded from every structure-level analysis.

---

## 2. What atom embeddings encode

**Question.** Do atom embeddings group by chemical element, by local environment, or by structure?

**Method** (`embeddings.atom_element_clustering`). All 93,633 atoms of 5,000 random sr_1m structures
(85 elements). For each atom, the ten nearest neighbours in embedding space (Euclidean) vote for an
element; the score is the fraction of atoms whose vote is correct. η²(element) is the fraction of
feature variance explained by element identity.

**Results.**

| Space | 10-NN element agreement |
|---|---|
| Majority-class baseline | 4% |
| UMA, PCA 2D | 12% |
| UMA, UMAP 2D | 29% |
| UMA, raw 128D | 63% |
| UMA, PCA 25D | 81% |
| UMA, standardised 128D | 83% |
| MH1, raw 96D | 92% |

* Element identity explains 79% of the variance of the MH1 readout features and about 26% for UMA.
  MH1 is element-dominated; UMA is organised mainly by local environment.
* A few high-variance UMA channels track the OMat24 subset label. Per-channel standardisation removes
  this and raises UMA agreement from 63% to 83%.
* Only 0.3% of neighbours are atoms of a different element from the same structure, so atom
  embeddings do not cluster by structure.

**Two-dimensional maps** (`embeddings.umap_atoms`, `embeddings.plot_umap_comparison`,
`embeddings.plot_umap_families`). cuML UMAP (26.06) of all 18.7M sr_1m atoms for each model with
identical settings: 15 neighbours, min_dist 0.1, spectral initialisation, NN-descent graph
(4 clusters, overlap 2), 200 epochs, seed 42, raw features, Euclidean metric. MH1 forms separate
islands, each owned by one or a few elements (2D element agreement 86%). UMA forms one connected
manifold in which element families occupy overlapping bands ordered by coordination environment
(29%). Refitting the UMA map on OMat24 atoms only, instead of Aryan's joint fit that included
external data, gives 30%. Each fit takes about 7 minutes on one A100.

---

## 3. MACE feature variants

**Question.** Is the element dominance specific to the MH1 readout, or common to all MACE layers?

**Method.** Per-atom invariant descriptors were extracted with mace's
`MACECalculator.get_descriptors(invariants_only=True, num_layers=-1)` in float64
(`embeddings.extract_mace_atom_variants`): both interaction layers of MACE-MH1 (omat_pbe head;
512 + 512) and of MACE-OMAT-0 (128 + 128), for 50,188 sr_1m structures (927,070 atoms: 50,000 random
structures with at least two atoms plus the 188 benchmark frames of §5). Their Z-weighted sums
reproduce the archive's per-structure MH1-1024 and OMAT0-128 vectors to a median relative error of
2.7 × 10⁻⁸, so the descriptors are identical to those used to build the archive.

Eight variants were compared (`embeddings.evaluate_mace_variants`, `embeddings.umap_variants`), all
standardised per channel: UMA-128, MH1-96 readout, MH1-16 omat_pbe head (channels 80–95), MH1-512
layer 1, MH1-512 layer 2, MH1-1024 both layers, OMAT0-128 layer 2, OMAT0-256 both layers.

**Results.**

| Variant | 10-NN agreement, full space | 10-NN agreement, 2D UMAP | η²(element) |
|---|---|---|---|
| UMA-128 | 82% | 41% | 0.25 |
| MH1-96 readout | 93% | 81% | 0.41 |
| MH1-16 omat_pbe head | 89% | 73% | 0.40 |
| MH1-512 layer 1 | 100% | 99% | 0.81 |
| MH1-512 layer 2 | 100% | 98% | 0.68 |
| MH1-1024 both | 100% | 98% | 0.74 |
| OMAT0-128 layer 2 | 99% | 97% | 0.53 |
| OMAT0-256 both | 99% | 97% | 0.57 |

* Every interaction-layer invariant is effectively an element label: in 2D each becomes about one
  compact island per element, with environment variation confined inside the islands.
* The readout hidden units are the MACE features that retain the most environment information. The
  16 omat_pbe units behave like the full readout.
* UMA remains the only representation with a connected 2D manifold. Its 2D agreement here (41%) is
  higher than on the 18.7M-atom map (29%) because these features were standardised.

The effect of the variant on atom matching is reported in §6.

---

## 4. Joint atom maps of five datasets

**Question.** Where do atoms of other datasets lie relative to OMat24, and which physical
attributes organise the maps?

**Method** (`embeddings.joint_atom_umap`, `embeddings.atom_attributes`, `embeddings.plot_joint_umaps`).
One UMAP per model (same settings as §2) over 20.9M atoms: all OMat24 sr_1m atoms plus the 10% atom
samples of MOF-off (1.05M atoms), MAD (0.26M), MP-ALOE (0.59M) and MatPES (0.31M). AM has no MH1
atom bank and was excluded; 557 MAD atoms with non-finite MH1 features were dropped from both maps.
Per-atom attributes were read from the source structures: DFT force magnitude |F|, atoms per
structure, and volume per atom (undefined for MAD's non-periodic molecules and clusters, 4.5% of its
atoms). A single-atom MAD frame has no rows in the bank and was skipped.

**Enrichment measure.** On a 300 × 300 pixel grid, a dataset's atoms are counted in pixels where
that dataset is at least ten times more abundant, relative to its size, than OMat24.

| Dataset | UMA | MH1 |
|---|---|---|
| MOF-off | 93% | 83% |
| MAD | 42% | 43% |
| MatPES | 11% | 14% |
| MP-ALOE | 7% | 10% |

**Results.**

* MOF-off atoms lie predominantly in regions OMat24 samples sparsely: the carbon and hydrogen
  ribbons in UMA and a few non-metal islands in MH1. MP-ALOE and MatPES lie almost entirely inside
  the OMat24 footprint; MAD is intermediate.
* The six OMat24 generation protocols (AIMD 1000 K, AIMD 3000 K, rattled relax, rattled 300, 500
  and 1000) populate nearly the same regions. Rattled 300/500/1000 frames leave some regions empty
  that AIMD and rattled-relax frames fill, consistent with the bond distortion found in §9.
* Colouring pixels by the median percentile of each attribute (percentiles over all 20.9M atoms, so
  that the 90% OMat24 share does not compress the scale) shows smooth gradients across the whole
  UMA manifold, most clearly in volume per atom; in MH1 the same gradients appear inside element
  islands. Large cells, mainly MOF-off (median 170 atoms per structure), are distinct in both.
* Median |F|: 1.1 eV/Å in OMat24, 1.8 eV/Å in MOF-off (off-equilibrium by construction), 0.5–0.6 eV/Å
  in MAD, MP-ALOE and MatPES.

---

## 5. Comparing structures of different cell size

**Question.** How can two structures with different numbers of atoms be compared, and how is
correctness verified?

### Benchmark (`matching.build_similar_structures`)

OMat24 derives several structures from each Alexandria parent. The rattled-relax structure uses a
primitive-sized cell (2–8 atoms); the rattled-300 structure uses a 2× or 4× supercell with random
displacements (median 0.6 Å from ideal sites). 1,212 materials have both. A pair was retained when
the volume per atom agrees within 5% and pymatgen's `StructureMatcher` (ltol 0.3, stol 0.5,
angle_tol 8°, supercells allowed) maps the supercell onto the small cell; the alignment provides the
ground-truth correspondence of every supercell atom. Ten curated pairs were used for development
and 84 held-out pairs for evaluation; 76 of 160 held-out candidates could not be aligned and were
excluded.

### Retrieval with structure vectors (`matching.compare_structures`)

Each of 300 random small cells must rank its own supercell first among 300 supercells.

| Representation | UMA | MH1 |
|---|---|---|
| Stored sum H | 4% | 2% |
| H / ΣZ | 64% | 61% |
| Mean of atom vectors | 22% | 61% |
| Optimal transport between atom sets (weights 1/N) | 51% | 81% |

Extensive sums fail. Division by the total atomic number is the simplest correction and the best
for UMA; optimal transport between the atom sets is the best for MH1. Over all 1,212 pairs the task
is harder because many supercells are also strained (H / ΣZ: 57% UMA, 50% MH1).

---

## 6. Atom correspondence between a cell and its supercell

**Question.** Which atom of the small cell corresponds to each atom of the supercell, and does the
matching respect local structure?

**Metrics.** Correct element; exact site (the true partner); and neighbour preservation, the
fraction of neighbour pairs in the supercell (distance < 1.3σ, σ the median nearest-neighbour
distance) whose matched partners are also neighbours.

### Embedding-only optimal transport (`matching.ot_atom_mapping`)

OT between the two sets of atom embeddings (squared Euclidean cost, uniform weights, the small cell
tiled to the supercell's size) assigns 98–100% of atoms to the correct element, finds the exact site
for 62% (UMA) and 53% (MH1) of atoms against 49% for random assignment within each element, and
preserves 50–54% of neighbour pairs, indistinguishable from random (51%). OT has no geometric term.
In BaLiTb every error is a symmetry-equivalent site; in KLiCl₂ inequivalent K and Cl sites are
exchanged over 3–4 Å.

### Fused Gromov–Wasserstein (`matching.fgw_atom_mapping`)

FGW adds a term that penalises mapping neighbouring atoms to non-neighbouring atoms:

min over T of (1 − α) Σᵢⱼ Mᵢⱼ Tᵢⱼ + α Σᵢⱼₖₗ (C₁[i,k] − C₂[j,l])² Tᵢⱼ Tₖₗ

where M is the squared embedding distance, T the transport plan, and C a Gaussian neighbour kernel
exp(−d²/2σ²) within each structure. Three modifications were required for periodic crystals:

| Modification | Problem addressed | Effect |
|---|---|---|
| Tile the small cell with a lattice-only integer supercell matrix | a cell and its supercell have inconsistent kernels: each neighbour appears once in the small cell but is distributed over images in the supercell | the objective prefers the true correspondence in 10/10 development pairs (previously 3/10) |
| Sum the kernel over all periodic images | minimum-image distances captured only 23–66% of each neighbourhood in thin cells | MH1 exact-site rate 45% → 55% |
| 100 random same-element initialisations | the objective is non-convex; two starts stayed at the OT solution | neighbour preservation close to the true correspondence |

α = 0 recovers OT; α = 0.1–0.25 performed best on the development pairs. The solver is POT 0.9.7.

### Held-out evaluation (`matching.compare_mapping_methods`)

On the 84 held-out materials, with parameters fixed on the development pairs, differences were
tested per material with bootstrap confidence intervals and Holm-corrected Wilcoxon signed-rank
tests.

| Method | Neighbour pairs preserved |
|---|---|
| Embedding-only GW (UMA / MH1 / UMA+MH1) | 50–51% |
| Random assignment within element | 51% |
| OT (UMA / MH1 / UMA+MH1) | 51–52% |
| FGW (UMA / MH1 / UMA+MH1) | 68–69% |
| True correspondence (ceiling) | 74% |

* Significant: FGW preserves 16–18 points more neighbour pairs than OT, and is better for 75–79 of
  the 84 materials.
* Significant: embedding-only GW is worse than OT by 12–15 points in exact-site rate.
* Not significant: α = 0.1 versus 0.25, and UMA versus MH1 versus their concatenation
  (each block scaled by 1/√width).
* 4× pairs are easier (81–86% exact site) than 2× pairs (57–66%), because fewer same-element sites
  compete.

### Effect of the MACE variant (`embeddings.evaluate_mace_variants --stage match`)

All eight variants of §3 on the 94 benchmark pairs, with OT and with FGW (periodic kernel, α = 0.25,
30 initialisations):

| Variant | Correct element OT / FGW | Exact site OT / FGW | Neighbours preserved OT / FGW |
|---|---|---|---|
| UMA-128 | 96.9 / 97.6 | 67.3 / 69.9 | 52.3 / 67.2 |
| MH1-96 readout | 98.7 / 98.9 | 68.7 / 70.6 | 52.1 / 68.3 |
| MH1-16 omat_pbe head | 97.1 / 98.4 | 66.7 / 67.9 | 51.1 / 68.1 |
| MH1-512 layer 1 | 100 / 100 | 69.0 / 70.6 | 52.6 / 68.4 |
| MH1-512 layer 2 | 100 / 100 | 70.6 / 73.8 | 50.4 / 68.5 |
| MH1-1024 both | 100 / 100 | 69.9 / 71.5 | 53.6 / 69.0 |
| OMAT0-128 layer 2 | 99.5 / 99.7 | 70.6 / 70.5 | 51.8 / 68.4 |
| OMAT0-256 both | 99.7 / 100 | 68.4 / 70.5 | 51.6 / 68.0 |

Element-pure layers remove the remaining element errors, but exact-site rates change little.
Relative to the MH1 readout, no variant differs significantly in exact-site rate after Holm
correction for seven comparisons (largest: MH1 layer 2, +3 points, p = 0.07).

### Further observations

* **Embedding-only Gromov–Wasserstein** (`matching.gw_embedding_mapping`), using intra-structure
  embedding distances as C and no M term, has nothing linking elements across the two structures and
  exchanges entire elements (BaLiTb across models: 0% correct element; the GW objective prefers an
  incorrect assignment in 49 of 50 fits).
* **Symmetry** is lost in the rattled data: all rattled-300 supercells are P1 at 0.5 Å tolerance, and
  only 15–19% of small cells retain the parent space group. Symmetry-equivalent assignments are
  therefore scored only where the parent group was recovered.
* **Cost** per pair on one core: OT 0.1 ms; one FGW solve 0.3 ms; 102 initialisations about 30 ms,
  plus 40–60 ms for kernel construction.
* An interactive viewer of the OT, FGW and GW assignments in 3D is produced by
  `matching.export_mapping_viewer` from `viewer/atom_mapping_viewer.template.html`.

---

## 7. A size-intensive structure distance

**Question.** For arbitrary structure pairs, where one cannot be tiled into the other, how can FGW
be made independent of cell size?

**Method** (`structure_distance.structure_fgw`, `structure_distance.validate_structure_fgw`).
The periodic kernel summed over all images is multiplied by the number of atoms, K = N · Σ over
images exp(−d²/2σ²), so that a supercell's kernel equals the primitive cell's on average. Feature
and kernel scales (m₀, k₀) are fixed globally so that every distance in a matrix is comparable.
Features are UMA and MH1, each normalised by its own width; σ = 3 Å; α = 0.5.

**Validation.** Each of the 94 small cells must retrieve its own 2× or 4× supercell among 94.

| Setting | Top-1 retrieval |
|---|---|
| σ = 1 Å | 39% |
| σ = 2 Å | 79% |
| Features only (α = 0) | 95% |
| σ = 3 Å, α = 0.5 | 97% |

A narrow kernel is too sensitive to rattling; at σ = 3 Å the geometric term adds to the feature term.

---

## 8. MOF-off versus OMat24

### Composition and atom-level overlap (`embeddings.embedding_overlap`)

Only 2 of 2,552 MOF-off r²SCAN compositions occur in a 10M-structure OMat24 sample (a praseodymium
formate and an iron oxalate). 99.4% of MOF-off structures contain both C and H, against 0.11% of
OMat24 structures; median sizes are 116 and 12 atoms. All MOF-off elements occur in OMat24, but
OMat24 carbon and hydrogen are almost exclusively in inorganic environments.

An atom is "outside" OMat24 if its nearest OMat24 atom is farther than the 99th percentile of
OMat24 nearest-neighbour distances (20,000 atoms per set). In UMA space this holds for 68% of MOF-off
C, 28% of N, 22% of Cu, 5% of H and 3% of O atoms, about one third of all MOF-off atoms; in MH1, which
is element-dominated, for 0.3%, 17%, 2%, 0.1% and 0%.

### Pairwise structure distances (`structure_distance.build_fgw_sample`, `compute_pairwise_fgw`, `analyze_pairwise_fgw`)

The size-intensive FGW distance of §7 over 100 MOF-off and 1,000 random OMat24 structures
(1,100 × 1,100 matrix, α = 0 and 0.5):

* MOF-off forms a separate block. Median distances: MOF–MOF 1.06, OMat24–OMat24 0.41, MOF–OMat24 1.60.
  98 of 100 MOFs are closer to another MOF than to any OMat24 structure.
* Random OMat24 structures differ mainly by composition, so the generation protocols are hard to
  distinguish, except that AIMD 3000 K frames are slightly displaced and are the protocol closest to
  MOF-off.

**Closest pairs** (`structure_distance.plot_closest_pairs`). The FGW plan shows where the atoms of
each MOF element are sent. The closest pair overall is a Cu–I framework and H₁₂CuI₆ (I → I 100%,
Cu → Cu 68%). The only C- and H-containing OMat24 structure in the sample, an organic crystal
C₃H₅N₅O, is the nearest neighbour of 80 of the 100 MOFs. Elements absent from the OMat24 structure
are assigned to the most similar available element (for Ba(OH)₂: N and O to O, S to Ba).

---

## 9. Search of 10M OMat24 structures for MOF analogues

**Question.** Which OMat24 structures are closest to MOF-off frameworks, and from which generation
protocol do they come?

### Queries

Ten MOF-off frameworks with ten different metals (Cu, Zn, Co, Cd, Fe, Mn, Ag, Ni, Ce, La), 42–276
atoms, against the OMat24 sr_10m sample minus the 12,275 unphysical frames.

### Two-stage search (`mof_search.prefilter`, `mof_search.rerank`)

Exact FGW for 10M structures and 10 queries would cost about 360 core-hours. A surrogate score ranks
all structures first: OT between the atom feature sets plus a one-dimensional Wasserstein distance
between the per-atom neighbour-density distributions (the "eccentricity" of each atom under the
intensive kernel), combined into one score. On a check set the surrogate correlates with FGW
at Spearman ρ = 0.90 and places 99% of the FGW top 10 within its own top 10%. The top 50,000
candidates per query (0.5%; 168,736 distinct structures) were re-ranked with exact FGW (UMA features,
σ = 3 Å, α = 0.5, global constants m₀ = 253.5 and k₀ = 18.13): 500,000 distances. The prefilter cost 46 core-hours over 400 Slurm tasks. For 7 of
10 queries, every FGW top-20 match came from prefilter rank ≤ 520; for the three largest MOFs
(202–276 atoms) some came from ranks of 11,000–35,000, and their best matches are poor in any case.

### Results

* **About half of the queries have chemically consistent analogues**
  (`mof_search.plot_search_matches`). The Co MOF matches a manganese coordination compound
  (Co → Mn 100%; ligand C, N and O onto C, N and O/S). A manganese formate matches a lithium manganese
  oxalate (Mn → Mn 95%, O → O 100%).
* **Failure mode.** The Zn MOF's best match is a 3000 K hydrocarbon frame: 80% of the MOF atoms are C
  or H and every atom carries equal weight, so the metal, N and O are assigned to H or C. The Ag, Ni
  and Ce MOFs match only large organosilicon or organometallic cells.
* **Every query's closest match is a near-equilibrium AIMD frame** (`mof_search.best_per_subset`).
  Relative to each query's best match overall, rattled-relax matches are 9–43% farther and often
  chemically consistent (a copper thiocyanate-type compound for the Cu MOF); rattled 300/500/1000
  subsets return small metallic cells 1.2–1.9 times farther, and restricting them to C- and
  H-containing structures does not close the gap.
* **Mechanism** (`mof_search.compare_subset_terms`). With composition controlled (only OMat24
  structures containing C and H), AIMD and rattled-relax frames are equally close to the MOFs on
  features alone (median 1.65–1.68), whereas rattled 300/500/1000 frames are 25–30% farther
  (2.06–2.20). Rattling displaces atoms independently and stretches C–H bonds: 62–64% of C–H bonds in
  rattled 300/500/1000 frames deviate from 1.09 Å by more than 0.15 Å, against 6–20% in AIMD and 7% in
  rattled relax. The three rattling labels distort hydrogen to the same extent, and UMA encodes the
  distorted environments as different.
* **Grouped matrices** (`structure_distance.build_grouped_samples`, `plot_grouped_matrices`). Balanced
  420 × 420 matrices (60 structures per group; random, and organic-only) with the search's global
  constants. Among organic frames, rattled 300/500/1000 cluster together regardless of label. With
  geometry included, AIMD 3000 K is the protocol closest to MOF-off, largely because its large, dense
  cells resemble MOF cells.

### Global structure vectors versus FGW neighbours (`mof_search.global_vs_fgw`)

**Question.** For the queries whose FGW neighbours were chemically meaningful (Q6 Mn formate,
Q3 Co, Q4 Cd, Q1 Cu), would global structure vectors have found the same neighbours?

**Method.** Global vectors pool the same z-scored UMA atom features that FGW uses, over all 10M
sr_10m structures: the Z-weighted mean (the archive's H/ΣZ after per-channel standardisation) and
the plain mean; the global distance is Euclidean. Unphysical frames are excluded as in the search.
For each query: rank agreement with FGW over the 50,000 reranked candidates, the global rank (over
10M) of FGW's top-10 distinct neighbours, and the global method's own neighbours (top-10 distinct
overall, top-5 distinct per family in AIMD 1000 K, AIMD 3000 K and rattled relax), each scored
with exact FGW and the same verdict rules as before.

| Query | Spearman with FGW (Z-mean / mean) | Overlap of top-10 (Z-mean / mean) | Median global rank of FGW top-10 (Z-mean) | Sensible, top-10 overall (FGW / Z-mean / mean) | Sensible, per family of 15 (FGW / Z-mean / mean) |
|---|---|---|---|---|---|
| Q6 Mn | 0.30 / 0.26 | 0.1 / 0.0 | 27 | 10 / 10 / 8 | 15 / 13 / 10 |
| Q3 Co | 0.11 / 0.14 | 0.1 / 0.5 | 19 | 5 / 2 / 4 | 9 / 4 / 8 |
| Q4 Cd | 0.11 / 0.04 | 0.8 / 0.6 | 6 | 6 / 2 / 5 | 8 / 3 / 6 |
| Q1 Cu | −0.11 / 0.23 | 0.0 / 0.0 | 823,460 | 5 / 0 / 1 | 5 / 0 / 2 |

* Global vectors rank structures very differently from FGW: within the reranked candidates the rank
  correlation is 0.30 at most and negative for Q1. (The candidates were preselected by an OT-based
  prefilter, which narrows the range and lowers these correlations.)
* For small, homogeneous MOFs (the Mn formate) global vectors recover chemically sensible neighbours
  almost as well as FGW, although at larger FGW distance (median 1.25× the best match for the
  Z-weighted mean against 1.14× for FGW's own top-10).
* For the larger, chemically mixed MOFs, global neighbours are mostly "partly" sensible: they share
  the ligand elements but put the metal on Mg, Na, Li, P or H (for example C₈MgN₆ and C₄H₃NaO₅ for
  the Co and Cd MOFs). The plain mean does better than the Z-weighted mean here, probably because
  Z-weighting lets the heavy atoms dominate the average.
* For Q1 (Cu) the two methods disagree completely. FGW's neighbours rank around 800,000th by global
  distance, and the Z-weighted mean's nearest structures are small rattled-relax intermetallic cells
  (BrMoRh, BMoSe₂W, AgITc₃) at 1.7–2.2× the FGW best distance; averaging presumably discards the Cu
  coordination environment that FGW matches.
* Conclusion: global vectors are a usable first filter for simple frameworks, but they do not
  reproduce FGW's chemically meaningful neighbours for mixed-ligand MOFs; the atom-level geometry
  term is what makes the FGW matches meaningful.

Outputs: `data/mof_search/global_vs_fgw/` (`pairs.csv` with every pair and its verdict, `summary.csv`,
`verdicts_per_family.csv`, `scatter.png`, `verdicts.png`). Pooled vectors and query distances for all
10M structures: `$SCRATCH/global_pool/`.

### Deliverables

* `data/mof_search/aimd_best_matches.csv`: the best AIMD 1000 K and AIMD 3000 K match for each of the
  10 MOFs (20 rows) with written chemical assessments (7 sensible, 6 partly, 7 not).
* `data/mof_search/mof_neighbours_per_subset.csv`: the five nearest neighbours per MOF in AIMD 1000 K,
  AIMD 3000 K and rattled relax (150 rows) with rule-based assessments from the FGW plans (element
  mapping, metal and donor-atom checks): 52 sensible, 65 partly, 33 not.
  (`mof_search.aimd_best_match_facts`, `mof_search.mof_neighbours_per_subset`).

---

## 10. Conclusions, limitations and next steps

### Conclusions

1. Geometry is required for atom correspondence: FGW with a periodic neighbour kernel preserves local
   connectivity, whereas embeddings alone do not. The choice of embedding is secondary.
2. Structure comparison across cell sizes requires an intensive representation: H/ΣZ, optimal
   transport, or the size-intensive FGW distance.
3. MACE interaction-layer invariants are effectively element labels; for matching, the layer is
   immaterial. UMA embeddings carry environment information that MACE features largely do not.
4. MOF-off is chemically disjoint from OMat24; its closest OMat24 analogues are near-equilibrium AIMD
   frames.
5. Rattled 300/500/1000 frames distort C–H bonds independently of their nominal temperature.

### Limitations

* The matching benchmark is small (94 materials), which limits the power to separate embeddings.
* Ground truth relies on StructureMatcher alignment; 76 of 160 held-out candidates could not be
  aligned, which may bias the benchmark towards simpler materials.
* Symmetry-equivalent sites are recognised only where the parent space group survives rattling.
* Uniform atom weights let abundant C and H dominate the transport cost for MOFs.
* The prefilter misses some top matches for the largest MOFs.
* The global-versus-FGW comparison covers four queries; it should be repeated over more MOFs.

### Next steps

* Penalise metal-to-non-metal assignments, or weight metal centres, in the transport cost.
* Extend the matching benchmark to several hundred materials.
* Score symmetry-equivalent assignments against ideal Alexandria parent structures.
* Use global vectors as a cheaper first-stage filter only for small, homogeneous frameworks.
* Obtain access to the 468 restricted files.

---

## 12. Test-time adaptation by fine-tuning on FGW neighbours

**Question.** Does fine-tuning MACE-OMAT-0 on the 20 OMat24 structures closest to a target MOF
(by structure FGW) improve its accuracy on that MOF?

**Method** (`tta.build_sets`, `tta.prepare`, `tta.evaluate`, `tta.compare`; `slurm/tta_finetune.slurm`).
Targets are the four search queries with meaningful FGW neighbours (Q6 Mn formate, Q3 Co, Q4 Cd,
Q1 Cu). Training: OMat24 sr_10m frames ranked 1–20 by FGW, with their OMat24 PBE(+U) labels;
validation: ranks 21–25 (the target is never used for training or model selection). Controls: 20
random OMat24 frames, and the 20 nearest by the Z-weighted global vector. Fine-tuning follows
Aryan's MACE fine-tuning method (group-internal, `/scratch/gpfs/ROSENGROUP/shared/aryan/finetune_method`):
energy/force/stress loss 1/10/10, lr 1e-4, EMA 0.99, AMSGrad, float32, 60 epochs, batch 4, E0s of the
foundation model, seed 123, one GPU (about 3 minutes per run). The target set is every MOF-off PBE
frame of the framework (4 frames for Q6, 39–40 for the others, 300–1000 K), scored before fine-tuning
and after epochs 1, 2, 3, 6, 11, 21, 41 and 60. Before training, the trainer's validation errors equal
those of the foundation model.

MOF-off's PBE set comes from a D3BJ dataset without +U; OMat24 is PBE(+U) without dispersion. The
target energy error therefore carries a constant per-atom offset (170–240 meV/atom) that OMat24
data cannot remove; forces and offset-free energies are the meaningful measures.

**Target force MAE (eV/Å), before and after 60 epochs:**

| Target | Before | FGW neighbours | Global-vector neighbours | Random structures |
|---|---|---|---|---|
| Q6 Mn | 0.223 | 0.247 (+11%) | 0.222 (−1%) | 0.250 (+12%) |
| Q3 Co | 0.220 | 0.215 (−2%) | 0.218 (−1%) | 0.244 (+11%) |
| Q4 Cd | 0.156 | 0.139 (−11%) | 0.141 (−10%) | 0.169 (+8%) |
| Q1 Cu | 0.245 | 0.235 (−4%) | 0.256 (+5%) | 0.282 (+15%) |

**Results.**

* Fine-tuning on random OMat24 structures always degrades the target (+8 to +15% force error):
  the model specialises to unrelated chemistry.
* Fine-tuning on neighbours avoids that degradation and gives modest gains for three of four
  targets, largest for the Cd MOF (−11%) even though Cd is absent from its training structures. FGW
  and global-vector neighbours perform similarly; FGW is better for the Cu MOF, the global vector for
  the Mn formate (−9% after 6 epochs, back to −1% by 60).
* Offset-free energy errors fall for Q3 with FGW (8.5 → 6.4 meV/atom) and Q1 (32.1 → 29.2), and change
  little otherwise. FGW and global neighbours also reduce the raw energy offset for Q3 and Q4
  (for example 172 → 72 meV/atom for Q4); random structures do not.
* Choosing the epoch by validation error (target-blind) keeps most of the gain for Q3 and Q4, where
  the best validation epoch is the last; for Q6 and Q1 it stops after 1–4 epochs, with little change.
* Caveats: one seed and one set of hyperparameters; the FGW top-20 consists of AIMD frames of only
  5–8 materials; the targets' metals are absent from the training structures for Q3 and Q4; the
  labels of training and target differ in dispersion and +U treatment.
* Not pursued: adding a D3(BJ) correction to put target energies at the reference's level of theory.
  Raw energy errors therefore mix a genuine model correction (the foundation model is already
  67–68 meV/atom off on the OMat24 neighbours of Q3 and Q4, and fine-tuning removes most of that) with
  the dispersion offset; only forces and offset-free energies are interpreted.

**Round 2: all ten queries, 100 training structures, lower learning rates.** The same protocol on all
ten search queries (adding Q2 Zn, Q5 Fe, Q7 Ag, Q8 Ni, Q9 Ce, Q10 La; 4–90 PBE frames each), with
training sets of 100 structures (ranks 1–100, validation 101–105; the 20-structure sets are their first
20) and learning rates 3e-5 and 1e-5 (20 structures). The optimiser is Adam (AMSGrad); the plateau
scheduler (patience 50) never lowers the rate within 60 epochs, so each run uses a constant rate.
120 runs in total; `slurm/tta_batch.slurm` chains runs within one GPU job.

Target force MAE, change after 60 epochs relative to MACE-OMAT-0, over the ten targets:

| Training set | lr | Mean | Median | Targets improved | Range |
|---|---|---|---|---|---|
| 20 nearest by FGW | 1e-4 | +1.2% | +0.1% | 5/10 | −10.6 to +11.6% |
| | 3e-5 | −0.1% | −0.3% | 5/10 | −9.7 to +7.2% |
| | 1e-5 | −0.6% | −0.7% | 6/10 | −8.0 to +6.4% |
| 100 nearest by FGW | 1e-4 | +1.7% | +1.5% | 4/10 | −13.7 to +12.6% |
| 20 nearest by global vector | 1e-4 | +4.2% | +0.7% | 5/10 | −10.0 to +27.7% |
| | 3e-5 | +0.8% | −1.7% | 6/10 | −10.2 to +18.0% |
| | 1e-5 | −1.9% | −2.5% | 6/10 | −9.2 to +8.0% |
| 100 nearest by global vector | 1e-4 | +5.6% | +2.8% | 4/10 | −8.5 to +29.4% |
| 20 random | 1e-4 | +9.6% | +10.7% | 0/10 | +2.8 to +15.1% |
| | 3e-5 | +3.7% | +3.2% | 2/10 | −3.4 to +11.5% |
| | 1e-5 | +2.2% | +1.2% | 5/10 | −3.5 to +9.9% |
| 100 random | 1e-4 | +13.3% | +8.8% | 0/10 | +0.3 to +51.4% |

* Averaged over ten targets, fine-tuning on neighbours does not improve the target: mean changes lie
  within ±2% at every learning rate, with about half the targets better and half worse. Random
  structures are consistently worse than neighbours at the same learning rate.
* The four queries chosen for meaningful neighbours benefit most (Q4 Cd −8 to −14% with FGW, Q3 and
  Q1 −2 to −5%); the six added queries mostly do not (Q7, Q8, Q9 worse with most sets).
* 100 structures do not help on average: they improve Q6 (−5%) and Q4 (−14%) but degrade Q7, Q8 and
  Q10. Ranks 21–100 are further from the target, and 100 structures mean five times more updates.
* Lower learning rates shrink every change towards zero and remove the large late degradations
  (global vector Q2 +28% → +8%, FGW Q5 +12% → +3%), at the cost of smaller gains (FGW Q4 −11% → −8%).
  Global-vector neighbours at 1e-5 have the best average (−1.9%, worst case +8%).
* Stopping at the best scored epoch, which requires the target, would improve 8–9 of 10 targets with
  neighbour sets (mean −2 to −4%). The validation error on the held-out neighbours does not find
  that epoch reliably, so this is an upper bound, not an achievable result.
* Offset-free energy errors change by less than 1–2 meV/atom in most cases, with no consistent
  direction across learning rates.

**Which FGW term predicts the gain** (`tta.neighbour_terms`). Each training structure is rescored against the
query frame and FGW is split at the optimal plan into the feature term ⟨M,T⟩ (UMA feature cost of
matched atoms) and the geometry term (GW loss on the periodic kernels); the recomputed FGW equals the
search value to 1e-16. Averaged over each training set and compared with the change in target force
MAE (Spearman ρ; positive = a larger term goes with a larger error increase):

| Sets | lr | FGW | feature term | geometry term | same-element share |
|---|---|---|---|---|---|
| 30 sets of 20 (all methods) | 1e-4 | +0.45 | **+0.60** | +0.33 | −0.50 |
| 20 neighbour sets of 20 (FGW, global) | 1e-4 | +0.33 | **+0.58** | +0.23 | −0.34 |
| 20 neighbour sets of 20 | 1e-5 | +0.05 | **+0.38** | −0.03 | −0.32 |
| 30 sets of 100 | 1e-4 | +0.55 | +0.42 | **+0.56** | −0.36 |

* For 20-structure sets the feature term is the best predictor: sets whose atoms carry UMA features
  close to the target's improve it (all sets with mean feature term below 0.9 improve or stay within
  +5%). The geometry term adds little at 20 structures and nothing at low learning rate.
* With 100 structures the geometry term becomes predictive as well: the extra ranks bring in
  geometrically different structures (Q7, Q8, Q10), and those runs degrade.
* The global vector often finds sets with a lower feature term but a higher geometry term than FGW
  (Q6, Q9, Q7, Q8); where its feature advantage is large (Q9: 0.77 vs 1.32) it fine-tunes better.
  FGW at α = 0.5 weighs geometry equally, which can favour geometrically similar but chemically
  less similar structures.
* Within a target, the set (FGW or global) with the lower term wins only 5–7 of 10 times, so the
  correlation mainly separates easy targets from hard ones rather than choosing between two sets.
  Ten targets and one seed; the correlations are indicative.

**Round 3: 30 targets, feature-weighted FGW, seeds, training variants.**

*Setup.* Twenty further targets (`tta.targets`: MOF-off sample structures that are not search
queries, ≥ 20 PBE frames, ≤ 200 atoms, at most two per metal; 18 metals, 26–186 atoms, 37–90 frames)
join the ten search queries. Training sets are re-ranked by FGW at α = 0.25 and α = 0.1 (more weight on
features) within a candidate pool (`tta.select_alpha`: top 1000 by FGW at α = 0.5 plus top 500 by the
global vector for the queries; top 1000 by the global vector for the new targets, whose α = 0.5 sets are
therefore pool-restricted). Each target is fine-tuned on five 20-structure sets (FGW α 0.5 / 0.25 / 0.1,
global vector, random) at lr 3e-5, 60 epochs. All runs go through one GPU job at a time
(`slurm/tta_batch.slurm`: up to 8 runs share the GPU when ≥ 12 GB is free; chained jobs skip finished
runs). The loss is MACE's energy/force/stress MSE (per-atom energy, weights 1/10/10); with OMat24's
typical errors the force term is ~99% of it and the stress term (eV/Å³) is negligible.

*Change versus MACE-OMAT-0 after 60 epochs, 30 targets, lr 3e-5* (`tta.effect`; mean, 95% bootstrap
interval over targets, targets improved):

| Metric | FGW α 0.5 | FGW α 0.25 | FGW α 0.1 | Global vector | Random |
|---|---|---|---|---|---|
| Force MAE | +1.4% [−1, +5], 13/30 | +0.6% [−2, +4], 16/30 | +0.3% [−2, +3], 14/30 | +1.7% [−2, +5], 16/30 | +5.3% [+3, +8], 6/30 |
| Force MAE, query frame | +1.6%, 10/30 | +0.9%, 12/30 | +0.4%, 14/30 | +2.3%, 13/30 | +6.8%, 7/30 |
| Energy MAE, raw | −19%, 27/30 | −24%, 26/30 | −26%, 27/30 | −24%, 27/30 | +1.5%, 14/30 |
| Energy MAE, offset-free | −2.7%, 19/30 | −4.1%, 19/30 | −1.9%, 17/30 | −2.7%, 19/30 | +6.8%, 9/30 |
| Stress MAE | −10%, 21/30 | −16%, 22/30 | −13%, 22/30 | −11%, 22/30 | +39%, 8/30 |

* Neighbours do not reduce the force or frame-to-frame energy error relative to the foundation model;
  they reduce stress (−10 to −16%) and the raw energy (offset-dominated).
* Neighbours are clearly better than random structures (FGW α 0.1 vs random: force −5.0 points,
  23/30, p = 0.002; raw energy −28, p < 0.001; stress −52, p < 0.001) but not better than the global
  vector on any metric (α 0.5 is worse on raw energy, p = 0.02). α 0.1 beats α 0.5 on raw energy
  (−7 points, p = 0.003), not on forces.
* +U labels: OMat24 uses PBE+U for oxides/fluorides of Co, Cr, Fe, Mn, Mo, Ni, V, W; MOF-off has no +U.
  On the 10 targets with such a metal every set degrades forces (+2.4 to +5.8%); on the other 20,
  neighbour sets are neutral (−0.8 to −0.1%) and random sets cost 6.5%. Within the +U group the share
  of +U-labelled training structures does not track the change.
* Seeds: repeating 29 runs with seed 7 changes the target errors by 0.1–0.3 points (correlation 1.00);
  per-target differences are properties of the target and training set, not training noise. The
  remaining seed repeats were dropped.

*Along training and stopping rules* (`tta.epochs`). The training-set force error falls 40–48% over
60 epochs and the held-out neighbours' 5–22%, but the target's force error is lowest after 1–6 epochs
and drifts up afterwards. Stopping at the lowest held-out-neighbour error does not find the target's
best epoch (it picks epoch ~60 for neighbour sets). Best fixed schedules on the 10 search targets:
FGW α 0.1 at lr 3e-5 for 1–6 epochs (−1.7 to −1.8%, 9/10), or at lr 1e-5 for 40–60 epochs (−2.0%, 8/10;
stress −21%). Stopping at the target's own best epoch (an upper bound) gains −2 to −4%.

*Training variants* (10 search targets, 30 epochs; `tta.epochs --sweep`, FGW α 0.25 set):

| Protocol | Epochs | Force MAE | Energy, offset-free | Stress |
|---|---|---|---|---|
| all weights, lr 3e-5 | 6 / 60 | −1.2% / −0.9% | −0.7% / −1.1% | −21% / −14% |
| readouts only, lr 1e-3 | 6 / 30 | −1.0% / −0.2% | −1.7% / +0.3% | +11% / +7% |
| products + readouts, lr 1e-4 | 6 / 30 | −0.9% / −1.3% | −2.2% / −4.8% | +1% / −4% |
| LoRA rank 4, lr 1e-3 | 6 / 30 | −0.6% / 0.0% | −1.4% / +2.7% | −16% / −20% |
| stress weight 1000, lr 3e-5 | 6 / 30 | −1.3% / −1.7% | −2.9% / −3.2% | +0.3% / +10% |

No variant changes the force error by more than ~1 point relative to training all weights (all paired
p > 0.16), and none protects against random structures (+4 to +6% after 30 epochs). Freezing layers
loses the stress improvement, and weighting stress 100× makes the target's stress worse (22–29 points
worse than default, p ≈ 0.02): OMat24 stresses lack the D3 dispersion pressure present in the MOF-off
reference.

*Which FGW term predicts the gain, 30 targets* (`tta.neighbour_terms`). All 150 round-3 training sets
are rescored with FGW at the search's α = 0.5 and split into feature and geometry terms, as above.
The terms are related to the change after 60 epochs at lr 3e-5 in two ways. *Across sets* compares
every (target, set) pair and mixes target difficulty with set quality. *Within target* subtracts each
target's mean over its sets, so only the choice of training set for one target is compared.

| Spearman ρ with % change, neighbour sets (120; 30 targets) | FGW | feature term | geometry term | same-element share | +U share |
|---|---|---|---|---|---|
| force MAE, across sets | +0.19 | **+0.52** | 0.00 | −0.35 | +0.41 |
| force MAE, within target | −0.09 | +0.14 | −0.10 | −0.08 | −0.10 |
| energy offset-free, across sets | +0.03 | +0.06 | +0.02 | −0.18 | −0.13 |
| stress MAE, across sets | +0.09 | +0.19 | +0.03 | −0.17 | +0.05 |
| stress MAE, within target | −0.11 | +0.22 | −0.11 | −0.09 | +0.11 |

(|ρ| ≥ 0.19 is p < 0.05 at n = 120.)

* **The feature term identifies targets that neighbours help.** At the target level (mean over a
  target's four neighbour sets) the feature term and the force change have ρ = +0.62. On targets
  without a +U metal the feature term averages 0.77 and forces change by −0.5%. On targets with a +U
  metal it averages 1.20 and forces change by +3.9%. Excluding the +U targets, ρ is still +0.36
  (p = 0.001). OMat24 has no close, plain-PBE chemistry for these metals.
* **No term picks the best set for a given target.** Within a target, no term predicts force or
  offset-free energy (|ρ| ≤ 0.14). The set with the lowest feature term gives the lowest force error
  for 9 of 30 targets, where chance is 7.5. The one weak within-target signal is stress: sets with
  lower feature terms improve stress a little more (ρ = +0.22, p = 0.02).
* The geometry term predicts nothing at 20 structures. The +U share of a set correlates with its
  feature term (ρ = +0.50), so the two cannot be separated across targets.
* Offset-free energy is not predicted by any term, in either comparison.

**Round 4: FGW α 0.25 on 79 targets.** The round-3 protocol for FGW α 0.25 (20 structures, all
weights, lr 3e-5, 60 epochs, scored at epochs 6 and 60) is applied to 49 further targets
(`tta.targets --extend`). These are every remaining MOF-off sample structure with ≥ 20 PBE frames,
≤ 200 atoms and its own framework, without the per-metal cap: 18 Cu, 11 Zn, 5 Co, 3 each of Fe, Mn,
Ag and Cd, one each of In, Ni and Yb, 26–186 atoms. Pools and sets are built as for the round-3
S targets. Only the α 0.25 set is fine-tuned, so the comparison is with MACE-OMAT-0 and not with
random or global sets (`tta.effect --cohorts fgwmid`).

*Change versus MACE-OMAT-0, FGW α 0.25* (mean, 95% bootstrap interval, targets improved):

| Targets | Force MAE, 6 ep. | Force MAE, 60 ep. | Energy offset-free, 6 / 60 ep. | Stress, 6 / 60 ep. | Energy raw, 60 ep. |
|---|---|---|---|---|---|
| all 79 | **−0.9%** [−1.8, −0.0], 48/79 | −0.5% [−2.1, +1.1], 43/79 | **−4.3%** [−7.1, −1.5] / −1.6% | −13% / −14% | −27% |
| 49 new (round 4) | −1.5% [−2.6, −0.4], 30/49 | −1.3% [−3.2, +0.6], 27/49 | −4.3% / 0.0% | −14% / −14% | −28% |
| 30 earlier | −0.0%, 18/30 | +0.6%, 16/30 | −4.3% / −4.1% | −10% / −16% | −24% |
| 57 without a +U metal | −1.3% [−2.4, −0.3], 38/57 | −1.6% [−3.6, +0.5], 36/57 | −4.0% / −1.5% | −14% / −14% | −31% |
| 22 with a +U metal | +0.2%, 10/22 | **+2.2%** [+0.1, +4.3], 7/22 | −5.1% / −1.7% | −10% / −15% | −14% |

* With 79 targets the small gains become statistically detectable. After 6 epochs the force error
  falls by 0.9% (Wilcoxon p = 0.003) and the offset-free energy error by 4.3% (p = 0.003). After 60
  epochs neither is significant (p = 0.39 and 0.09), because longer training drifts away from the
  target, as in round 3. Stress improves at both points (p < 10⁻⁵).
* The new targets, mostly Cu and Zn, behave like the non-+U targets of round 3: forces fall by
  1–2% (Cu −2.0%, Zn −1.5% after 6 epochs). Targets with a +U metal do not improve on forces and get
  worse with longer training (+2.2% after 60 epochs; Co +2.8%, Ni +3.6%, Mn +1.4%).
* The size of the effect is unchanged: about −1 to −2% on forces, −4% on offset-free energy and
  −13% on stress, best after a short fine-tune.

Round-4 outputs: `effect/cohorts_fgwmid.csv` (per target), `cohorts_fgwmid_summary.csv`,
`cohorts_fgwmid.png`.

**Later work** (adapt-or-not rule `tta.abstain`, weight interpolation `tta.interpolate`, round 5 with
MOF-off neighbours `tta.select_mof` and +U-free OMat24 neighbours, MOF-off controls) is summarised in
[REPORT.md §5.6–5.8](REPORT.md#56-adapt-or-not-a-target-blind-rule-ttaabstain); how to rerun everything is in
[TTA_REPRODUCE.md](TTA_REPRODUCE.md).

*Conclusion.* With 20 OMat24 structures selected by FGW and labelled at OMat24's level of theory,
fine-tuning MACE-OMAT-0 changes the target MOF's force and frame-to-frame energy errors by at most
~2%, whatever the selection weight, training length or adaptation method; the gains are bounded by
the level-of-theory mismatch and the distance between OMat24 neighbours and MOFs, not by the
optimisation. FGW (or global-vector) selection matters in that it avoids the degradation caused by
random data. Best protocol found: FGW α 0.1–0.25, all weights, lr 3e-5 for ~6 epochs (or lr 1e-5),
giving about −1.5 to −2% on forces and −20% on stress. On 79 targets (FGW α 0.25, 6 epochs) the
gains are small but significant: forces −0.9%, offset-free energy −4.3%, stress −13%. They come
from targets without a +U metal (forces −1.3%). The FGW feature term predicts which targets
benefit, but no FGW term picks the best training set for a given target.

Outputs: `data/tta/summary.csv` (every scored epoch), `data/tta/final.csv` (before vs after),
`target_force_mae.png` (20 vs 100 structures), `target_force_mae_lr.png` (learning rates),
`convergence*.png` (training loss, validation and target errors per epoch), `final_force_change.png`,
`neighbour_terms/` (FGW terms per training structure, `correlations.csv` across sets and within
target, `terms_vs_gain.png` at lr 3e-5, `terms_vs_gain_force.png` at other learning rates and sizes),
`alpha_rank/` (pool rankings), `effect/` (all metrics vs controls, +U split, `cohorts_fgwmid*` for
round 4), `epochs/` (stopping rules, `sweep*`,
`protocol_metrics.csv`); runs in `$SCRATCH/tta/`.

---

## 11. Where each result comes from

| Result | Module(s) | Output |
|---|---|---|
| Element agreement and η² (§2) | `embeddings.atom_element_clustering` | `runs/atom_element_clustering/` |
| Atom UMAPs (§2) | `embeddings.umap_atoms`, `embeddings.plot_umap_comparison`, `embeddings.plot_umap_families` | `runs/atom_umap_sr1m/` |
| MACE variants (§3, §6) | `embeddings.extract_mace_atom_variants`, `embeddings.evaluate_mace_variants`, `embeddings.umap_variants` | `$SCRATCH/mace_variants/`, `data/mace_variants/` |
| Joint maps (§4) | `embeddings.joint_atom_umap`, `embeddings.atom_attributes`, `embeddings.plot_joint_umaps` | `$SCRATCH/joint_umap/`, `runs/joint_umap/` |
| Benchmark and retrieval (§5) | `matching.build_similar_structures`, `matching.compare_structures` | `data/similar_structures{,_heldout}/` |
| OT, FGW, GW matching (§6) | `matching.ot_atom_mapping`, `matching.fgw_atom_mapping`, `matching.gw_embedding_mapping`, `matching.compare_mapping_methods` | `data/similar_structures_heldout/comparison/` |
| Structure FGW validation (§7) | `structure_distance.validate_structure_fgw` | `data/structure_fgw/validation/` |
| MOF-off vs OMat24 (§8) | `embeddings.embedding_overlap`, `structure_distance.*` | `data/overlap/`, `data/structure_fgw/` |
| 10M search (§9) | `mof_search.*`, `structure_distance.build_grouped_samples` | `data/mof_search/`, `data/grouped_fgw/` |
| Global vectors vs FGW (§9) | `mof_search.global_vs_fgw` | `$SCRATCH/global_pool/`, `data/mof_search/global_vs_fgw/` |
| Test-time adaptation (§12) | `tta.*` | `$SCRATCH/tta/`, `data/tta/` |

Job scripts for each stage are listed in the [README](../README.md#pipeline).
