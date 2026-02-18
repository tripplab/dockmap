#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

dockmap -v \
  --cache-surface quicksurf_cache_01.npz \
  --out-prefix docking_map_example_01 \
  --protein data/target.pdb \
  --peptides data/peps.pdb \
  --scores data/vina_score.dat \
  --ppi-file data/target_relevant_residues.txt \
  --ppi-footprint residue_points --ppi-footprint atom_contour \
  --background curvature \
  --background-smooth 1 \
  --background-colorbar \
  --background-colorbar-mode raw \
  --background-colorbar-location right \
  --trace-pose 5 \
  --pose-label topN \
  --pose-label-top 5 \
  --seam-rotate -120 \
  --radius-scale 4.0 \
  --grid-spacing 0.4 \
  --surface-quality max \
  --ppi-near-surface-eps 12.0 \
  --pose-layer scatter \
  --map equirect
