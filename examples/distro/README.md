# Distro examples (shared dataset)

This folder contains two distributable `dockmap` examples that both use the same input files in `data/`:

- `data/target.pdb`
- `data/peps.pdb`
- `data/vina_score.dat`
- `data/target_relevant_residues.txt`

## Prerequisites

From repository root:

```bash
python -m pip install .



Expected stdout/log highlights
Example 01:
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

[INFO] dockmap.cli: dockmap pipeline start
[INFO] dockmap.cli: Load protein atoms ...
[INFO] dockmap.cli: Load protein atoms done (0.01s)
[INFO] dockmap.cli: Load peptide poses + scores ...
[INFO] dockmap.cli: Load peptide poses + scores done (0.00s)
[INFO] dockmap.cli: Loaded 5 peptide poses
[INFO] dockmap.cli: Parse PPI residue list ...
[INFO] dockmap.cli: Parse PPI residue list done (0.00s)
[INFO] dockmap.cli: Loaded 17 PPI residues
[INFO] dockmap.cli: Protein chains present: X(1935 atoms)
[INFO] dockmap.cli: PPI chains in contacts file: X
[INFO] dockmap.cli: PPI residues: 17  | matched: 17  | missing: 0
[INFO] dockmap.cli: Build QuickSurf mesh ...
[INFO] dockmap.cli: Build QuickSurf mesh done (41.55s)
[INFO] dockmap.cli: Surface mesh: 1032936 vertices, 344312 faces
[INFO] dockmap.cli: Project peptide centers to surface + map to spherical UV ...
[INFO] dockmap.cli: Project peptide centers to surface + map to spherical UV done (13.41s)
[INFO] dockmap.cli: Mapped poses: 5
[INFO] dockmap.cli: Seam rotation: user -> -120.00 deg
[INFO] dockmap.cli: Map PPI footprint to UV ...
[INFO] dockmap.cli: Map PPI footprint to UV done (51.36s)
[INFO] dockmap.cli: Mapped PPI contour points: 139
[INFO] dockmap.cli: Mapped PPI residue points: 17
[INFO] dockmap.cli: Compute background scalar (curvature) ...
[INFO] dockmap.cli: Background smoothing: 1 iterations
[INFO] dockmap.cli: Compute background scalar (curvature) done (37.95s)
[INFO] dockmap.cli: Render 2D map ...
[INFO] dockmap.cli: Render 2D map done (2.88s)
[INFO] dockmap.cli: Wrote map: docking_map_example_01.png
[INFO] dockmap.cli: Write CSV outputs ...
[INFO] dockmap.cli: Wrote CSV: docking_map_example_01_poses_mapped.csv
[INFO] dockmap.cli: Wrote CSV: docking_map_example_01_clusters.csv
[INFO] dockmap.cli: Wrote CSV: docking_map_example_01_ppi_contour_mapped.csv
[INFO] dockmap.cli: Wrote CSV: docking_map_example_01_ppi_residue_points_mapped.csv
[INFO] dockmap.cli: Write CSV outputs done (0.00s)
[INFO] dockmap.cli: dockmap pipeline complete



Expected stdout/log highlights
Example 02:
dockmap -v \
  --cache-surface quicksurf_cache_02.npz \
  --out-prefix docking_map_example_02 \
  --protein data/target.pdb \
  --peptides data/peps.pdb \
  --scores data/vina_score.dat \
  --ppi-file data/target_relevant_residues.txt \
  --ppi-footprint residue_points --ppi-footprint atom_contour \
  --background radial \
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
  --pose-layer trace

[INFO] dockmap.cli: dockmap pipeline start
[INFO] dockmap.cli: Load protein atoms ...
[INFO] dockmap.cli: Load protein atoms done (0.01s)
[INFO] dockmap.cli: Load peptide poses + scores ...
[INFO] dockmap.cli: Load peptide poses + scores done (0.00s)
[INFO] dockmap.cli: Loaded 5 peptide poses
[INFO] dockmap.cli: Parse PPI residue list ...
[INFO] dockmap.cli: Parse PPI residue list done (0.00s)
[INFO] dockmap.cli: Loaded 17 PPI residues
[INFO] dockmap.cli: Protein chains present: X(1935 atoms)
[INFO] dockmap.cli: PPI chains in contacts file: X
[INFO] dockmap.cli: PPI residues: 17  | matched: 17  | missing: 0
[INFO] dockmap.cli: Build QuickSurf mesh ...
[INFO] dockmap.cli: Build QuickSurf mesh done (37.91s)
[INFO] dockmap.cli: Surface mesh: 1032936 vertices, 344312 faces
[INFO] dockmap.cli: Project peptide centers to surface + map to spherical UV ...
[INFO] dockmap.cli: Project peptide centers to surface + map to spherical UV done (13.98s)
[INFO] dockmap.cli: Mapped poses: 5
[INFO] dockmap.cli: Seam rotation: user -> -120.00 deg
[INFO] dockmap.cli: Map PPI footprint to UV ...
[INFO] dockmap.cli: Map PPI footprint to UV done (49.73s)
[INFO] dockmap.cli: Mapped PPI contour points: 139
[INFO] dockmap.cli: Mapped PPI residue points: 17
[INFO] dockmap.cli: Compute background scalar (radial) ...
[INFO] dockmap.cli: Background smoothing: 1 iterations
[INFO] dockmap.cli: Compute background scalar (radial) done (22.99s)
[INFO] dockmap.cli: Project trace CA atoms to surface + map to spherical UV ...
[INFO] dockmap.cli: Project trace CA atoms to surface + map to spherical UV done (88.43s)
[INFO] dockmap.cli: Render 2D map ...
[INFO] dockmap.cli: Render 2D map done (1.23s)
[INFO] dockmap.cli: Wrote map: docking_map_example_02.png
[INFO] dockmap.cli: Write CSV outputs ...
[INFO] dockmap.cli: Wrote CSV: docking_map_example_02_poses_mapped.csv
[INFO] dockmap.cli: Wrote CSV: docking_map_example_02_clusters.csv
[INFO] dockmap.cli: Wrote CSV: docking_map_example_02_ppi_contour_mapped.csv
[INFO] dockmap.cli: Wrote CSV: docking_map_example_02_ppi_residue_points_mapped.csv
[INFO] dockmap.cli: Write CSV outputs done (0.00s)
[INFO] dockmap.cli: dockmap pipeline complete
