# mof_embeddings

Comparing crystal structures across datasets (OMat24, MOF-off, MAD, MP-ALOE, MatPES) with
UMA and MACE atom embeddings: what atom embeddings encode, how to match atoms between a cell
and its supercell with optimal transport and fused Gromov–Wasserstein (FGW), a size-intensive
structure distance, and a search of 10M OMat24 structures for analogues of MOF-off frameworks.

The methods and results are described in [`docs/ANALYSIS.md`](docs/ANALYSIS.md) and summarised in
[`slides/atom_matching_2026-10-26.pdf`](slides/atom_matching_2026-10-26.pdf).

## Layout

```
src/mof_embeddings/
  config.py              all input and output locations (override with MOF_EMB_* variables)
  embeddings/            per-atom embeddings: element structure, UMAPs, MACE variants, joint dataset maps
  matching/              cell-vs-supercell benchmark and atom matching (OT, FGW, GW), interactive viewer
  structure_distance/    size-intensive structure FGW, pairwise matrices, closest pairs
  mof_search/            two-stage search of OMat24 sr_10m for MOF-off analogues
slurm/                   job scripts (submit from the repository root); env.sh sets PYTHONPATH and interpreters
slides/                  build_deck.js (pptxgenjs) and make_assets.sh (redraws every deck figure)
viewer/                  HTML template of the atom-matching viewer
```

`data/`, `runs/` and `logs/` hold results on the cluster and are not tracked.

## Setup

The code runs on Princeton's Della cluster against the shared embedding archive
(`/scratch/gpfs/ROSENGROUP/shared/aryan/saved_embeddings`) and Aryan's processed structures.
Two existing conda environments are used read-only:

| Interpreter | Used for |
|---|---|
| `pretrain_analysis_env` (`$PY`) | everything else: numpy, torch, cuML, pymatgen, ase, scipy |
| `pretrain_analysis_env_mace` (`$PY_MACE`) | `embeddings.extract_mace_atom_variants` (mace 0.3.16; needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`) |

POT (optimal transport) is installed locally: `pip install --target .pydeps pot==0.9.7`.
Modules that need it call `config.use_local_packages()`.

```bash
source slurm/env.sh                        # PYTHONPATH=src, $PY, $PY_MACE, cd to the repository root
$PY -m mof_embeddings.matching.fgw_atom_mapping --help
sbatch slurm/run_python.slurm -m mof_embeddings.structure_distance.analyze_pairwise_fgw
```

## Pipeline

| Stage | Entry points | Job script |
|---|---|---|
| Element structure of atom embeddings | `embeddings.atom_element_clustering`, `embeddings.plot_atom_element_maps` | `run_python.slurm` |
| Atom UMAPs (18.7M OMat24 atoms) | `embeddings.umap_atoms`, `embeddings.plot_umap_comparison`, `embeddings.plot_umap_families` | `umap_atoms.slurm` |
| MACE feature variants | `embeddings.extract_mace_atom_variants`, `embeddings.evaluate_mace_variants`, `embeddings.umap_variants` | `mace_variants.slurm` → `mace_variant_eval.slurm` |
| Joint dataset maps | `embeddings.joint_atom_umap`, `embeddings.atom_attributes`, `embeddings.plot_joint_umaps` | `joint_umap.slurm`, `atom_attributes.slurm` |
| MOF-off vs OMat24 atom overlap | `embeddings.embedding_overlap` | `embedding_overlap.slurm` |
| Cell-vs-supercell benchmark | `matching.build_similar_structures`, `matching.compare_structures` | `run_python.slurm` |
| Atom matching | `matching.ot_atom_mapping`, `matching.fgw_atom_mapping`, `matching.gw_embedding_mapping`, `matching.compare_mapping_methods` | `mapping_comparison.slurm` |
| Matching figures and viewer | `matching.plot_ot_mapping`, `matching.plot_neighbour_preservation`, `matching.export_mapping_viewer` | — |
| Structure FGW distance | `structure_distance.structure_fgw`, `structure_distance.validate_structure_fgw` | `run_python.slurm` |
| Pairwise matrices | `structure_distance.build_fgw_sample`, `structure_distance.compute_pairwise_fgw`, `structure_distance.analyze_pairwise_fgw`, `structure_distance.build_grouped_samples`, `structure_distance.plot_grouped_matrices`, `structure_distance.plot_closest_pairs` | `pairwise_fgw.slurm`, `grouped_fgw.slurm` |
| 10M analogue search | `mof_search.prefilter`, `mof_search.rerank`, `mof_search.compare_subset_terms`, `mof_search.best_per_subset`, `mof_search.aimd_best_match_facts`, `mof_search.mof_neighbours_per_subset`, `mof_search.plot_search_matches`, `mof_search.plot_best_per_subset` | `mof_search.slurm` |
| Global vectors vs FGW neighbours | `mof_search.global_vs_fgw --stage pool / compare / plot` | `global_pool.slurm`, then `run_python.slurm` |

Every module documents its inputs, outputs and options in its docstring (`--help`).

## Slides

```bash
bash slides/make_assets.sh                 # redraw the figures into slides/assets_arial/
NODE_PATH=<node_modules with pptxgenjs> node slides/build_deck.js
```

`build_deck.js` applies the theme with the pptx skill's `apply_theme.js`; adjust that `require` path
if the skill lives elsewhere.
