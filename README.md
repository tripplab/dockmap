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
   - optionally recenter all angular coordinates on a selected pose with `--center-pose [N]`, using `--center-theta-phi THETA_DEG PHI_DEG` when the selected pose should land somewhere other than `0,0`
   - project `(theta, phi)` to a 2D map projection (`equirect`, `mollweide`, or `hammer`)
5. Convert the PPI residue list to a 2D footprint.
6. Create a 2D figure showing:
   - docking locations (scatter/hexbin/density/trace/centroid/md)
   - PPI overlay (points and/or contour)
   - optional background (radial or curvature-proxy relief)
7. Write CSV/JSON reports with mapped coordinates and alignment metrics.
8. Optionally export the computed surface mesh (OBJ/PLY/STL).

When `--pose-layer md` is selected, `dockmap` also treats the ligand poses like an ordered
trajectory: it computes one heavy-atom geometric center per ligand `MODEL`, connects those
COM points in PDB order on the map, marks the first pose with a filled triangle and the last
pose with a filled circle, and writes `_md` CSV/PNG diagnostics.

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

### Updating an already installed copy

If you already installed `dockmap` from a source checkout, update the checkout first and then
reinstall into the same environment.

#### pip environment

```bash
cd /path/to/dockmap
git pull
python -m pip install --upgrade .
```

If you installed in editable/developer mode, refresh dependencies and metadata with:

```bash
cd /path/to/dockmap
git pull
python -m pip install --upgrade -e .
```

#### micromamba environment

```bash
micromamba activate dockmap
cd /path/to/dockmap
git pull
python -m pip install --upgrade .
```

If optional raycast dependencies were installed previously and you want to keep them current:

```bash
micromamba activate dockmap
cd /path/to/dockmap
git pull
python -m pip install --upgrade ".[trimesh]"
```

After updating, confirm that the CLI on your `PATH` comes from the active environment:

```bash
which dockmap
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

Example centered run (place the first pose at `theta=0, phi=0`):

```bash
dockmap \
  --set s1:protein.pdb:peptide_poses.pdb:vina_scores.txt \
  --ppi-file ppi.txt \
  --center-pose \
  --out-prefix docking_map_centered
```

Use `--center-pose N` to center on a specific 1-based pose number, for example `--center-pose 5`.

Example trajectory-style COM run (`md` layer):

```bash
dockmap \
  --set traj:protein.pdb:ligand_models.pdb:vina_scores.txt \
  --ppi-file ppi.txt \
  --map mollweide \
  --pose-layer density \
  --pose-layer md \
  --out-prefix docking_map
```

This overlays a connected ligand COM trajectory on `docking_map.png`, using the same
`theta`/`phi` projection as the map. The first pose in the PDB `MODEL` order is marked with
a filled triangle, and the last pose is marked with a filled circle.

---

## Outputs

Typical outputs are:

- `docking_map.png` (or `pdf`/`svg`, depending on `--out-format`)
- `docking_map_poses_mapped.csv` (now includes `cluster_id`, ordered by cluster then score)
- `docking_map_clusters.csv` (cluster summary statistics)
- `docking_map_ppi_contour_mapped.csv` (when contour footprint is generated)
- `docking_map_ppi_residue_points_mapped.csv` (when residue-point footprint is generated)
- `docking_map_quicksurf.ply` (if `--export-mesh`)
- `docking_map_md.csv` (if `--pose-layer md`)
- `docking_map_md.png` (if `--pose-layer md`)


### `--pose-layer md`: ligand COM trajectory

Use `--pose-layer md` when the ligand poses PDB should be interpreted as an ordered series
of `MODEL` poses, such as frames or ordered docking/MD-like snapshots. This layer:

- computes one ligand COM per pose using the **geometric center of heavy atoms only**;
- preserves the pose order found in the PDB file;
- projects each COM through the same surface-map workflow used for other pose layers;
- draws a line connecting COMs on the existing map by `theta COM` / `phi COM`;
- color-codes MD COM points on the map by `--pose-layer-md-TL-distance` (default: `16` Å): green for `r COM <= threshold`, red for `r COM > threshold`;
- marks the first pose with a filled triangle and the last pose with a filled circle;
- writes `*_md.csv` and `*_md.png` outputs.

CSV columns in `*_md.csv` are:

```text
X COM,Y COM,Z COM,r COM,theta COM,phi COM,ro COM
```

Column meanings:

- `X COM`, `Y COM`, `Z COM`: heavy-atom ligand COM coordinates in the aligned/reference frame.
- `r COM`: distance from the protein center to the ligand COM, in Å.
- `theta COM`, `phi COM`: angular coordinates used by the map after seam rotation and any `--center-pose` offsets, written in radians.
- `ro COM`: angle in degrees between each pose COM vector and the first pose COM vector.

The `*_md.png` file contains two panels:

1. `r COM` vs pose number, with points color-coded by `--pose-layer-md-TL-distance` (default: `16` Å): green for `r COM <= threshold`, red for `r COM > threshold`.
2. `ro COM` vs pose number.

Pose number is the 1-based order found in the ligand PDB file.

### Cluster centroid vs PPI atom-contour annotation

`docking_map_clusters.csv` includes one additional column:

- `ppi_contour_level_index_max`

This value reports, for each cluster centroid, the **highest crossed contour level index**
for the PPI atom-contour field (`1..N`, where `N = --ppi-contour-levels`).

Rules:

- If the centroid is outside all contour levels, the field is empty (`NA`).
- If contour classification is unavailable (for example, no PPI atom contour points, or sparse PPI atom-cloud
  rendered as scatter fallback), the field is also empty (`NA`).
- Cluster row ordering is unchanged (still ranked by cluster size, then existing tie-breakers).

Contour classification is computed from the same histogram + blur + level construction used for rendering,
using the user-selected contour controls:

- `--ppi-contour-lon-bins`
- `--ppi-contour-lat-bins`
- `--ppi-contour-blur-sigma-px`
- `--ppi-contour-levels`

---

## Tips

### Headless nodes (no display)

If matplotlib complains about display/X11:

```bash
export MPLBACKEND=Agg
```

### Pose-centered angular origin (`--center-pose [N]`)

Use `--center-pose` to move the angular origin of the map to a selected peptide pose.
The optional `N` is a **1-based pose number**; if omitted, pose `1` is used.
After peptide centers are read and mapped to the surface, `dockmap` computes theta/phi offsets from the selected pose's peptide COM so that the selected pose is written and plotted at `theta=0, phi=0` by default.
Use `--center-theta-phi THETA_DEG PHI_DEG` with `--center-pose` to place the selected pose at a different target angular coordinate, specified in degrees.
The same offsets are then applied consistently to every angular layer: mapped poses, PPI contour and residue-point overlays, background mesh coordinates, trace lines, cluster centroids, and MD COM trajectories.

Examples:

```bash
# Center on the first pose
dockmap ... --center-pose

# Center on pose number 5
dockmap ... --center-pose 5

# Center on pose number 5 and place it at theta=45°, phi=90°
dockmap ... --center-pose 5 --center-theta-phi 45 90
```

`--center-pose` composes with `--seam-rotate`: seam rotation is calculated first, then the pose-centering offsets are added so the selected pose still lands at the requested angular target (`theta=0, phi=0` unless `--center-theta-phi` is provided).

### Pose clustering

Pose clustering uses **spherical angular distance** on `(theta, phi)` (great-circle distance on the unit sphere).
By default, mapped angular coordinates in CSV outputs are written in **radians**.
Exception: in `docking_map_clusters.csv`, `theta_centroid` and `phi_centroid` are written in **degrees**.

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
