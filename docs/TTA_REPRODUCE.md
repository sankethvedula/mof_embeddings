# Reproducing the test-time adaptation experiments

For Aryan. This guide reruns the fine-tuning experiments of [REPORT.md §5](REPORT.md#5-test-time-adaptation-tta)
on Della and proposes the next two steps: many more targets, and UMA instead of MACE. Results and
numbers are in REPORT.md; methods in detail are in [ANALYSIS.md §12](ANALYSIS.md#12-test-time-adaptation-by-fine-tuning-on-fgw-neighbours).

## What the experiment does

For one **target** MOF (a MOF-off framework):

1. **Select** 25 structures close to the target by structure FGW on UMA atom features. Ranks 1–20 are
   for training and 21–25 for validation. They come from OMat24 sr_10m or from MOF-off frames of
   other frameworks.
2. **Fine-tune** MACE-OMAT-0 on the 20 structures with your fine-tuning settings, for 60 epochs, with
   a checkpoint kept at several epochs.
3. **Score** the foundation model and each kept checkpoint on every MOF-off PBE frame of the target
   framework (energy, forces, stress). The target is never used for training or model selection.
4. **Compare** the change against MACE-OMAT-0 over many targets, against control selections (random,
   global vector), with bootstrap intervals and paired Wilcoxon tests.

## 1. Setup

```bash
git clone git@github.com:sankethvedula/mof_embeddings.git
cd mof_embeddings
source slurm/env.sh        # sets PY, PY_MACE (your conda envs), PYTHONPATH; run from the repo root
```

* **Environments.** `PY` = `pretrain_analysis_env` (selection, analysis). `PY_MACE` =
  `pretrain_analysis_env_mace` (LMDB writing, evaluation, `mace_run_train` from your
  `pretrain_analysis/mace-src` fork). Both default to your env paths in `slurm/env.sh`.
* **Foundation model.** Copy `mace-omat-0-medium.model` from your `finetune_method` folder to
  `finetune/mace-omat-0-medium.model`. It is gitignored, as is `finetune/aryan_method/`.
* **Output root.** Set `export MOF_EMB_SCRATCH=/scratch/gpfs/ROSENGROUP/<you>`. Runs go to
  `$MOF_EMB_SCRATCH/tta/` (about 300 GB for the 79-target study, mostly kept checkpoints) and pooled
  vectors to `$MOF_EMB_SCRATCH/global_pool/`. `MOF_EMB_ARCHIVE` (the embedding archive) and
  `MOF_EMB_PRETRAIN` (your `pretrain_analysis`) default to your paths.
* **Slurm account.** The scripts say `#SBATCH --account=guillermos`. Pass `--account=<yours>` to
  `sbatch` or edit the header. `tta_batch.slurm` runs on `gputest` when given `--time=00:59:00`.

## 2. Inputs that are not in git

`data/` is gitignored. Ask Sanketh for a copy of these files; his home and scratch are not readable
from rosengroup. Paths are relative to the repo.

| Input | Size | Used by | Rebuild instead with |
|---|---|---|---|
| `data/structure_fgw/sample/` (structures, UMA atom features of the 100 MOF-off + 1,000 OMat24 sample) | 31 MB | `tta.targets`, every selection | `structure_distance.build_fgw_sample` |
| `data/mof_search/queries.csv`, `constants.json` (FGW scale constants m₀, k₀, σ) | < 1 MB | all FGW scoring | `mof_search.prefilter` |
| `data/mof_search/rerank_parts/` (exact FGW, top 50,000 per query) | 12 MB | Q-target sets | `mof_search.rerank` |
| `data/mof_search/prefilter_parts/` (also holds the per-frame minimum distance used to drop the 12,275 unphysical frames) | 497 MB | `build_sets` | `mof_search.prefilter` (46 core-hours) |
| `data/tta/targets_extra.csv` (the 69 S targets) | small | everything | `tta.targets` (seeded) |
| `data/tta/alpha_rank/`, `data/tta/mof_rank/` (pool rankings) | 20 MB | `build_sets` | `tta.select_alpha`, `tta.select_mof` |
| `$MOF_EMB_SCRATCH/global_pool/` (Z-weighted mean UMA vector of all 10M sr_10m frames and distances to the queries) | 11 GB | `select_alpha`, `build_sets` | `sbatch slurm/global_pool.slurm` (30 min) |

With the copied `data/` files only `global_pool` has to be rebuilt. `select_alpha` writes
`plus_u.npy` there on first use.

## 3. Pipeline

All commands run from the repo root after `source slurm/env.sh`. Each step skips work that exists.

```bash
# 1. Targets: Q1-Q10 are the search queries; S targets are drawn from the MOF-off sample
#    (>= 20 PBE frames, <= 200 atoms). The 69 S targets of this study:
$PY -m mof_embeddings.tta.targets --n 20 --per-metal 2                 # round 3 (20 targets)
$PY -m mof_embeddings.tta.targets --extend --n 100 --per-metal 100     # round 4 (all remaining: 69)

# 2. OMat24 candidate pools and FGW rankings at alpha 0.5 / 0.25 / 0.1 (CPU)
$PY -m mof_embeddings.tta.select_alpha --workers 32 --skip-done
$PY -m mof_embeddings.tta.select_alpha --no-plus-u --targets S4 S12 S23 S82 S83   # pools without PBE+U

# 3. MOF-off candidates of other frameworks ranked by FGW alpha 0.25 (~1 min per target)
$PY -m mof_embeddings.tta.select_mof --workers 24 --skip-done

# 4. Target, training and validation LMDBs for every selection (MACE env: uses mace's LMDB writer)
$PY_MACE -m mof_embeddings.tta.build_sets
```

**Selections** (`tta.common.METHODS`; directory `$MOF_EMB_SCRATCH/tta/<target>/<selection>/`):

| Selection | Training structures |
|---|---|
| `fgw` | OMat24 nearest by FGW α 0.5 (the search's setting) |
| `fgwmid`, `fgwlow` | OMat24 nearest by FGW α 0.25 / 0.1 within the pool |
| `global` | OMat24 nearest by the Z-weighted mean UMA vector (control) |
| `random` | random OMat24 sr_10m frames (control) |
| `fgwmidnou` | as `fgwmid` without PBE+U-labelled structures |
| `moffgw` | MOF-off frames of other frameworks, FGW α 0.25, one per framework |
| `moffgwnf` | as `moffgw` without frameworks of the target's empirical formula |
| `mofglobal`, `mofrandom` | MOF-off frames of other frameworks by global vector / at random (controls) |

A selection identical to another one holds only `same_as.txt`. It is not trained, and the analysis
reuses the other selection's run.

```bash
# 5. Fine-tune and score. One GPU job runs up to PARALLEL runs at once; finished runs are skipped,
#    so chained jobs (afterany) continue where the previous one stopped.
#    Entry = TARGET:SELECTION[:LR[:SEED[:VARIANT]]]; TARGET:foundation scores MACE-OMAT-0 first.
RUNS="Q6:foundation Q6:fgwmid:3e-5 Q6:moffgw:3e-5 Q6:mofrandom:3e-5" PARALLEL=8 \
    sbatch --time=00:59:00 slurm/tta_batch.slurm
# a single run: TARGET=Q6 SELECTION=moffgw LR=3e-5 sbatch slurm/tta_finetune.slurm
```

* **Per run.** `slurm/tta_finetune.slurm` writes the `mace_run_train` config (`tta.prepare`) and
  trains for 60 epochs, about 3 minutes for 20 structures on an A100. It then scores epochs
  0, 1, 2, 5, 10, 20, 40 and 59 on the target, validation and training sets (`tta.evaluate`, float64)
  and keeps those checkpoints as `.model` files.
* **Learning rate.** All results in REPORT §5.4–5.8 use lr 3e-5.
* **Throughput.** About 40 runs fit in one 59-minute gputest job.

```bash
# 6. Weight interpolation w = (1-l) w_foundation + l w_finetuned (GPU; the kept models were saved on GPU)
TARGETS="$(cut -d, -f1 data/tta/targets_extra.csv | tail -n +2 | tr '\n' ' ') Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 Q10" \
    SELECTION=fgwmid ARGS="--lambdas 0 0.25 0.5 0.75 1" sbatch slurm/tta_interp.slurm

# 7. Analysis (CPU)
$PY -m mof_embeddings.tta.compare                        # data/tta/summary.csv, final.csv, figures
$PY -m mof_embeddings.tta.effect --lr 3e-5               # every metric vs controls, paired tests
$PY -m mof_embeddings.tta.effect --cohorts moffgw        # per-target table and cohort/metal summary
$PY -m mof_embeddings.tta.effect --interp fgwmid         # interpolation summary
$PY -m mof_embeddings.tta.abstain --method fgwmid        # adapt-or-not rules
$PY -m mof_embeddings.tta.epochs --lr 3e-5               # along-training curves and stopping rules
$PY -m mof_embeddings.tta.neighbour_terms terms --workers 10 && $PY -m mof_embeddings.tta.neighbour_terms relate
```

**Checks that should reproduce.**
* Before training, MACE's validation errors equal the foundation evaluation.
* The interpolation at λ = 0 and λ = 1 equals the stored foundation and fine-tuned evaluations.
* A second seed changes target errors by only 0.1–0.3 points.
* Round 4 (`fgwmid`, 79 targets, 6 epochs): forces −0.9%, offset-free energy −4.3%, stress −13%.

## Next steps for Aryan

### A. Many more targets

The current 79 targets use up the MOF-off structures of the 100-structure FGW sample that pass the
filters (≥ 20 PBE frames, ≤ 200 atoms). Going further needs targets from the MOF-off UMA bank
(`uma/atom/external_subsets/mof_off_r2scan`: 8,064 frames of 2,305 frameworks, all with a PBE
twin). Two functions need changing:

1. `tta.targets.candidates()`: draw frameworks from that bank instead of `data/structure_fgw/sample`.
   `tta.select_mof.bank()` already loads the bank with framework, atom offsets and source rows.
2. `tta.common.target_query()`: read the target's UMA atom features from the bank (rows
   `frame_atom_offsets[i]:[i+1]` of `atom_latents.npy`) and its structure from `R2SCAN_train.traj`,
   instead of the sample's `embeddings.npz` / `structures.extxyz`. `select_mof.rank_target` shows
   both reads.

Everything downstream works per target and is unchanged. Suggested scale:
* **Coverage:** 300–1,000 frameworks, stratified by metal, so that metal and +U cohorts have ≥ 30
  targets each.
* **Selections:** `moffgw`, `mofrandom`, `mofglobal`, `moffgwnf` and `fgwmid`.
* **Cost:** 5 runs per target, about 40 runs per GPU-hour.
* **CPU stages:** `select_alpha` and `select_mof` (about 1 minute per target) are per target and can
  run as Slurm arrays.
* **Duplicates:** with many targets from the same bank, exclude every frame of a target's framework
  from all other targets' MOF-off candidates. `select_mof` already does this per target. Also
  consider excluding same-formula frameworks (`moffgwnf`), because REPORT §5.8 shows that
  same-formula neighbours inflate the gain.

### B. The same protocol on UMA

Selection and analysis do not depend on the fine-tuned model. Selection already uses UMA atom
features, and the analysis reads per-frame CSVs. Only training and scoring are MACE-specific:

| Step | MACE now | UMA equivalent |
|---|---|---|
| Training data | ASE-LMDB written with mace's copy of fairchem's `LMDBDatabase` (`build_sets.write_lmdb`) | the same files should load with fairchem's `AseDBDataset`; otherwise write them with `fairchem.core.datasets` |
| Config + training | `tta.prepare` → `mace_run_train` (loss 1/10/10, EMA, AMSGrad, lr 3e-5, 60 epochs, batch 4, checkpoint every epoch) | a fairchem fine-tuning YAML for `uma-s-1p1`/`uma-s-1p2` with the same loss weights, lr, epochs and batch size, keeping checkpoints at epochs 1, 2, 3, 6, 11, 21, 41, 60 |
| Scoring | `tta.evaluate` with `MACECalculator` | the same function with `FAIRChemCalculator(predictor, task_name=...)`; the CSV columns (`set, frame, natoms, energy_ref, energy_pred, force_abs_sum, force_sq_sum, force_ref_rms, stress_mae`) must stay the same, so `tta.compare`, `effect`, `abstain`, `epochs` work unchanged |
| Interpolation | `tta.interpolate` (MACE state dicts) | the same formula on the UMA checkpoint's state dict |
| Run layout | `$MOF_EMB_SCRATCH/tta/<target>/<selection>/lr3e-05/eval_epoch<E>.csv` | the same layout under another root, or a `VARIANT` suffix (`tta.common.VARIANTS`), so foundation and fine-tuned files pair up |

Points to decide for UMA:

* **Task head.** UMA's `omat` task is OMat24's level of theory (PBE+U, no dispersion), which matches
  the OMat24 selections. The MOF-off labels are PBE-D3(BJ) without U. UMA's `odac` task (MOFs with
  dispersion) may already be closer. Score the foundation model with both heads before
  fine-tuning; check ODAC's functional and dispersion settings against MOF-off's first.
* **Comparison with MACE.** Fine-tune the `omat` head on the OMat24 selections, and the head
  closest to the target's labels on the MOF-off selections.
* **Fine-tuning scope.** UMA-S is mixture-of-linear-experts; fine-tuning all weights with 20
  structures may behave differently from MACE. Run the round-3 variant sweep (heads only vs all
  weights, two learning rates) on the 10 search targets before scaling up.
* **What to report.** The same tables as REPORT §5.5–5.8: % change in force, offset-free energy and
  stress MAE after 6 and 60 epochs. Do this per selection (OMat24 FGW, MOF-off FGW, controls) and per
  cohort (+U metal or not), with interpolation at λ 0.5.

### Recommended order

1. Reproduce round 4 on 10 targets (`fgwmid` + `foundation`) to check the setup against the numbers
   above.
2. Wait for the MOF-off controls (Sanketh, queued 12 Nov). If `moffgw` beats `mofrandom` and
   `mofglobal`, FGW selection matters at matched level of theory.
3. Scale to many targets with MACE (step A).
4. Port training and scoring to UMA and repeat on the same targets (step B).
