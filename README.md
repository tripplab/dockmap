# dockmap (v0.3.2)

Creative 2D “world map” of docking locations on a protein surface from many peptide poses, 
with an overlay of a known PPI region. The protein surface is computed internally 
(QuickSurf-like Gaussian density + isosurface extraction) with no external surface tools 
and no `scikit-image`.

---

## What it does

Given:

- a **protein** structure in PDB format
- a **peptide poses** PDB containing many poses (poses separated by `END`, `ENDMDL`, and/or `MODEL ... ENDMDL`)
- a **scores** text file (one Vina score per pose, same order as poses)
- a **PPI residue list** file (one residue per line)

`dockmap` will:

1. Build a **QuickSurf-like protein surface mesh** internally.
2. For each peptide pose:
   - compute the peptide center (COM or COG)
   - project it to the protein surface
   - map that surface point to spherical coordinates (θ, φ)
   - project (θ, φ) to a **2D map projection** (equirect / Mollweide / Hammer)
3. Convert the PPI residue list to a **2D footprint** (two selectable modes).
4. Create a **2D figure** showing:
   - docking locations (scatter/hexbin/density)
   - PPI overlay (points or contour)
   - optional background (radial or curvature “relief”)
5. Write CSVs with the mapped coordinates.
6. Optionally export the surface mesh (OBJ/PLY/STL). For PLY, you can optionally export a **per-vertex scalar** for coloring in ParaView/MeshLab.

---

## Installation

### Recommended (HPC-friendly): micromamba / conda-forge

Create an environment:

```bash
micromamba create -n dockmap -c conda-forge python=3.11 numpy matplotlib pip
micromamba activate dockmap

Optional (better surface projection / raycasting):

micromamba install -n dockmap -c conda-forge trimesh rtree

Install dockmap from the project directory (where pyproject.toml is):

pip install .

Check the CLI:

dockmap -h

Inputs
1) Protein PDB (--protein)

Single structure with ATOM/HETATM records.
2) Peptide poses PDB (--peptides)

Many poses in a single file. The parser is robust to either:

    END between poses

    ENDMDL between poses

    MODEL ... ENDMDL blocks

3) Vina scores (--scores)

Plain text file with one numeric score per pose, in the same order as the pose blocks in the peptide file.
4) PPI residue list (--ppi-file)

Example line format:

Chain  B  ILE    36
Chain  B  ALA    37
...

Only chain ID and residue number are used.
Quick start

Minimal run (2D map + CSVs):

dockmap \
  --protein protein.pdb \
  --peptides peptide_poses.pdb \
  --scores vina_scores.txt \
  --ppi-file ppi.txt \
  --out-prefix docking_map

A more “full” run (background + mesh export + vertex scalar)

dockmap \
  --protein protein.pdb \
  --peptides peptide_poses.pdb \
  --scores vina_scores.txt \
  --ppi-file ppi.txt \
  --map mollweide \
  --pose-layer density \
  --weight exp \
  --background curvature --background-smooth 5 \
  --export-mesh --mesh-format ply --mesh-vertex-scalar density \
  --out-prefix docking_map

Outputs:

    docking_map.png (or pdf/svg)

    docking_map_poses_mapped.csv

    docking_map_ppi_mapped.csv

    docking_map_quicksurf.ply (if --export-mesh)

Coloring the PLY scalar (ParaView)

If you export a PLY scalar (e.g., --mesh-vertex-scalar density or curv_proxy), the PLY will include an extra vertex property.

In ParaView:

    Open the .ply file.

    In the toolbar, set Coloring to density (or your chosen --mesh-scalar-name).

    Rescale to data range (if needed) and choose a colormap.

MeshLab has similar “color by vertex scalar/quality” options.
Notes / tips
Headless nodes (no display)

If matplotlib complains about display/X11, force a non-interactive backend:

export MPLBACKEND=Agg

Optional trimesh

If you installed the optional extras (trimesh, rtree), you can use:

    --pose-projection raycast (center → COM direction)
    Otherwise, it falls back to nearest surface point.

Full CLI help

Run:

dockmap -h

