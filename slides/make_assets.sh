#!/bin/bash
# Redraw every figure used by build_deck.js into slides/assets_arial/ (Arial, no figure titles).
# Needs the results under data/ and runs/. Run from the repository root:  bash slides/make_assets.sh
set -euo pipefail
source slurm/env.sh
export MPLCONFIGDIR="$(mktemp -d)"
cat > "$MPLCONFIGDIR/matplotlibrc" <<'RC'
font.family: Arial, DejaVu Sans
mathtext.fontset: custom
mathtext.rm: Arial
mathtext.it: Arial:italic
mathtext.bf: Arial:bold
axes.unicode_minus: False
RC
A=slides/assets_arial; mkdir -p "$A"
run() { "$PY" -B -m "mof_embeddings.$1" "${@:2}"; }
run embeddings.plot_umap_comparison --run-dir runs/atom_umap_sr1m --no-title --out "$A/umap_comparison.png"
run embeddings.plot_umap_families --out "$A/umap_families.png"
run embeddings.plot_joint_umaps --no-title --out "$A"
"$PY" -B -c "
import json, mof_embeddings.embeddings.umap_variants as U
from mof_embeddings.embeddings.evaluate_mace_variants import OUT, Features
F = Features(); _, G = U.sample_labels(F, F.frames[:50000])
U.plot_compact(OUT / 'umap', G, json.loads((OUT / 'umap/knn_vote_2d.json').read_text()))"
cp data/mace_variants/umap/umap_variants_compact.png "$A/"
run matching.plot_ot_mapping --formulas BaLiTb KLiCl2 --per-row 1 --no-title --out "$A/ot_two.png"
run matching.plot_neighbour_preservation --formulas BaLiTb Ca2ZnH2 --no-title --out "$A/bonds_two.png"
run structure_distance.plot_grouped_matrices --matrix data/structure_fgw/pairwise/D_alpha0.5.npy \
    --sample data/structure_fgw/sample --out "$A/grouped_1100.png"
run structure_distance.plot_grouped_matrices --matrix data/grouped_fgw/organic/pairwise/D_alpha0.5.npy \
    --sample data/grouped_fgw/organic/sample --out "$A/grouped_organic_a05.png"
run structure_distance.plot_closest_pairs --rows 0 2 3 --no-title --out "$A/closest_pairs_3.png"
run mof_search.plot_search_matches --queries Q3 Q6 Q2 --no-title --out "$A/best_matches_3.png"
run mof_search.plot_best_per_subset --no-title --out "$A/best_per_subset.png"
"$PY" -c "
import json, glob, os
from PIL import Image
d = {os.path.basename(p): list(Image.open(p).size) for p in sorted(glob.glob('$A/*.png'))}
json.dump(d, open('$A/sizes.json', 'w'), indent=1)"
