# dockmap (v1.5.0)

Creative 2D “world map” of docking locations on a protein surface from many peptide poses,
with an overlay of a known PPI region.

The protein surface is computed internally (QuickSurf-like Gaussian density + isosurface extraction),
with no external surface tools and no `scikit-image` requirement.

---

## Methodology infographic

A compact workflow overview (input → transformations → output) can be generated locally:

```bash
python docs/generate_methodology_infographic.py
```

This command writes `docs/dockmap_methodology_infographic.png` in your working tree.

---

## What it does

Given one or more docking **sets** (repeatable `--set`), each containing:

- a **protein target** PDB
- a **peptide poses** PDB containing many poses (poses separated by `END`, `ENDMDL`, and/or `MODEL ... ENDMDL`)
- a **scores** text file (one Vina score per pose, same order as poses)
- a **PPI residue list** file (one residue per line)

`dockmap` will:

1. Load all sets.
2. Align target proteins to a reference set (`--reference-set`, default first set) using rigid Cα superposition
   and apply that transform to each set's ligand poses (alignment is skipped automatically for a single set, or with `--no-align`).
3. Build a QuickSurf-like protein surface mesh internally (reference frame).
4. For each peptide pose across all sets:
   - compute the peptide center (COM or COG)
   - project it to the protein surface
   - map that surface point to spherical coordinates `(theta, phi)`
   - project `(theta, phi)` to a 2D map projection (`equirect`, `mollweide`, or `hammer`)
5. Convert the PPI residue list to a 2D footprint.
6. Create a 2D figure showing:
   - docking locations (scatter/hexbin/density/trace)
   - PPI overlay (points and/or contour)
   - optional background (radial or curvature-proxy relief)
7. Write CSV/JSON reports with mapped coordinates and alignment metrics.
8. Optionally export the computed surface mesh (OBJ/PLY/STL).

---

## Requirements

- Python `>=3.9`
- Core Python dependencies (installed automatically):
  - `numpy>=1.22`
  - `matplotlib>=3.6`

Optional extras:

- `trimesh` + `rtree` (for `--pose-projection raycast`):
  - `pip install ".[trimesh]"`
- `scipy` (optional for workflows that may rely on SciPy-based analysis):
  - `pip install ".[scipy]"`

Or install all optional extras at once:

```bash
pip install ".[trimesh,scipy]"
```

---

## Installation

### Option 1: pip (from source checkout)

From the repository root (this folder, containing `pyproject.toml`):

```bash
python -m pip install --upgrade pip
python -m pip install .
```

For an editable developer install:

```bash
python -m pip install -e .
```

### Option 2: micromamba / conda-forge (HPC-friendly)

```bash
micromamba create -n dockmap -c conda-forge python=3.11 numpy matplotlib pip
micromamba activate dockmap
python -m pip install .
```

Optional raycast support:

```bash
python -m pip install ".[trimesh]"
```

Verify CLI install:

```bash
dockmap -h
```

---

## Input files

### 1) Docking set descriptor (`--set`) **required, repeatable**

Format:

```text
set_id:target.pdb:peptide_poses.pdb:vina_scores.txt
```

Where target and pose coordinates are PDB and scores are one numeric value per pose.

### 2) Peptide poses PDB

Multiple poses in one file; parser supports:

- `END` between poses
- `ENDMDL` between poses
- `MODEL ... ENDMDL` blocks

### 3) Score file

Plain text, one numeric score per pose, matching pose order in `--peptides`.

### 4) PPI residue file (`--ppi-file`) **required**

Example line format:

```text
Chain  B  ILE    36
Chain  B  ALA    37
```

Only chain ID and residue number are used.

---

## How to run

Minimal run (map + CSV outputs):

```bash
dockmap \
  --set s1:protein.pdb:peptide_poses.pdb:vina_scores.txt \
  --ppi-file ppi.txt \
  --cluster-distance 15 \
  --out-prefix docking_map
```

Example fuller run (density layer + curvature background + mesh export):

```bash
dockmap \
  --set ref:protein_ref.pdb:poses_ref.pdb:scores_ref.txt \
  --set alt:protein_alt.pdb:poses_alt.pdb:scores_alt.txt \
  --ppi-file ppi.txt \
  --reference-set ref \
  --map mollweide \
  --pose-layer density \
  --weight exp \
  --pose-density-sigma 1.2 \
  --background curvature --background-smooth 5 \
  --export-mesh --mesh-format ply --mesh-vertex-scalar density \
  --out-prefix docking_map
```

---

## Outputs

Typical outputs are:

- `docking_map.png` (or `pdf`/`svg`, depending on `--out-format`)
- `docking_map_poses_mapped.csv` (now includes `cluster_id`, ordered by cluster then score)
- `docking_map_clusters.csv` (cluster summary statistics)
- `docking_map_ppi_contour_mapped.csv` (when contour footprint is generated)
- `docking_map_ppi_residue_points_mapped.csv` (when residue-point footprint is generated)
- `docking_map_quicksurf.ply` (if `--export-mesh`)

---

## Tips

### Headless nodes (no display)

If matplotlib complains about display/X11:

```bash
export MPLBACKEND=Agg
```

### Pose clustering

Pose clustering uses **spherical angular distance** on `(theta, phi)` (great-circle distance on the unit sphere).
`theta` and `phi` are written in **radians** in CSV outputs.

Use `--cluster-distance` in **degrees** to set the clustering threshold (default: `15`).

### Pose density width (for `--pose-layer density`)

Use `--pose-density-sigma` to control how wide/smooth the density blobs are.

- Lower values (for example `0.8`–`1.2`) make density tighter around cluster members.
- Higher values (for example `2.5`–`4.0`) make broader, more diffuse blobs.

Example (tighter density):

```bash
dockmap ... --pose-layer density --pose-density-sigma 1.0
```

### Raycast projection

`--pose-projection raycast` requires trimesh support. If unavailable, use the default:

```text
--pose-projection nearest
```

### Full CLI help

```bash
dockmap -h
```
