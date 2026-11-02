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
* No analysis yet compares the neighbour sets given by global (structure-level) embeddings with those
  given by the atom-wise FGW distance.

### Next steps

* Penalise metal-to-non-metal assignments, or weight metal centres, in the transport cost.
* Extend the matching benchmark to several hundred materials.
* Score symmetry-equivalent assignments against ideal Alexandria parent structures.
* Compare k-nearest-neighbour sets from global embeddings (H/ΣZ) and from structure FGW.
* Obtain access to the 468 restricted files.

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

Job scripts for each stage are listed in the [README](../README.md#pipeline).
