from __future__ import annotations

# by trippm@tripplab.com [Feb 2026]

import argparse
import numbers
from pathlib import Path
import numpy as np
import csv
import json
from dataclasses import dataclass

from .util import (
    configure_logging,
    get_logger,
    Timer,
    write_mesh,
    vertex_laplacian_magnitude,
    radial_distance,
    normalize_01,
    smooth_scalar_neighbor_average,
)
from .io import (
    load_protein_atoms,
    load_poses,
    coords_from_atoms,
    center_of_geometry,
    center_of_mass,
    parse_ppi_file,
    protein_residue_inventory,
    validate_ppi_residues_exist,
    write_pdb_atoms,
    write_pdb_poses,
    AtomRecord,
    Pose,
)
from .surface import build_quicksurf_mesh, QuickSurfSpec, sample_field_trilinear
from .project import project_point_to_surface_nearest, project_point_to_surface_raycast
from .mapproj import surface_point_to_spherical_uv, auto_seam_rotation, apply_seam_rotation
from .ppi import ppi_residue_points_uv, ppi_atom_cloud_uv
from .clustering import cluster_connected_components, reorder_clusters_and_poses
from .viz import plot_map, PlotSpec
from . import __version__

log = get_logger(__name__)


@dataclass(frozen=True)
class DockSet:
    set_id: str
    protein_path: str
    peptides_path: str
    scores_path: str
    protein_atoms: list[AtomRecord]
    poses: list


def _set_usage_error(value: str) -> str:
    return (
        f"Invalid --set value: {value!r}. "
        "Expected format: set_id:target.pdb:poses.pdb:scores.txt"
    )


def _parse_set_arg(value: str) -> tuple[str, str, str, str]:
    # Split from the right to tolerate ':' in set_id.
    parts = value.rsplit(":", 3)
    if len(parts) != 4:
        raise ValueError(_set_usage_error(value))
    set_id, protein, peptides, scores = [p.strip() for p in parts]
    if not set_id or not protein or not peptides or not scores:
        raise ValueError(_set_usage_error(value))
    return set_id, protein, peptides, scores


def _extract_target_ca_by_residue(atoms: list[AtomRecord]) -> dict[tuple[str, int, str], AtomRecord]:
    out: dict[tuple[str, int, str], AtomRecord] = {}
    for a in atoms:
        if a.name.strip().upper() != "CA":
            continue
        key = (a.chain, int(a.resseq), a.icode or "")
        if key not in out:
            out[key] = a
    return out


def _kabsch_rigid_transform(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R, t so that mobile @ R + t best matches reference."""
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("Alignment requires Nx3 paired coordinates.")
    cm = mobile.mean(axis=0)
    cr = reference.mean(axis=0)
    xm = mobile - cm
    xr = reference - cr
    h = xm.T @ xr
    u, _, vt = np.linalg.svd(h)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = u @ vt
    t = cr - (cm @ r)
    return r, t


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    d2 = np.sum((a - b) ** 2, axis=1)
    return float(np.sqrt(np.mean(d2))) if d2.size else float("nan")


def _transform_atoms(atoms: list[AtomRecord], r: np.ndarray, t: np.ndarray) -> list[AtomRecord]:
    out: list[AtomRecord] = []
    for a in atoms:
        c = a.coord @ r + t
        out.append(
            AtomRecord(
                chain=a.chain,
                resname=a.resname,
                resseq=a.resseq,
                icode=a.icode,
                name=a.name,
                element=a.element,
                coord=np.array(c, dtype=float),
            )
        )
    return out


def _load_dock_sets(raw_sets: list[str]) -> list[DockSet]:
    sets: list[DockSet] = []
    seen: set[str] = set()
    for raw in raw_sets:
        sid, protein, peptides, scores = _parse_set_arg(raw)
        if sid in seen:
            raise SystemExit(f"Duplicate set_id in --set inputs: {sid!r}")
        seen.add(sid)
        protein_atoms = load_protein_atoms(protein)
        poses = load_poses(peptides, scores)
        if len(poses) == 0:
            raise SystemExit(f"No peptide poses loaded for set {sid!r}.")
        sets.append(
            DockSet(
                set_id=sid,
                protein_path=protein,
                peptides_path=peptides,
                scores_path=scores,
                protein_atoms=protein_atoms,
                poses=poses,
            )
        )
    return sets


def _format_csv_cell(value: object) -> object:
    """Format float-like CSV values to 3 decimals; leave other types unchanged."""
    if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        return f"{float(value):.3f}"
    return value


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("max_help_position", 34)
        kwargs.setdefault("width", 110)
        super().__init__(*args, **kwargs)


def _pose_id_to_ligid(pose_id: str) -> str:
    # pose0007 -> LIG0007
    digits = "".join([c for c in pose_id if c.isdigit()])
    if not digits:
        return f"LIG{pose_id}"
    return f"LIG{int(digits):04d}"


def _extract_ca_trace_atoms(peptide_atoms: list[AtomRecord]) -> list[AtomRecord]:
    """
    Return Cα atoms in peptide, in N->C order as they appear in the PDB.
    (We assume the peptide file is ordered; for cyclic peptides, this is still a consistent traversal.)
    """
    return [a for a in peptide_atoms if a.name.strip().upper() == "CA"]


def _select_pose_indices_for_trace(mode: str, scores: np.ndarray, nposes: int) -> list[int]:
    """
    Return 0-based pose indices selected by mode:
      - 'first'  -> [0]
      - 'best'   -> [argmin(scores)]
      - integer N (string) -> best N by score (ascending), e.g. '5' -> 5 best poses
    """
    m = str(mode).strip().lower()
    if nposes <= 0:
        return []
    if m == "first":
        return [0]
    if m == "best":
        return [int(np.argmin(scores))]

    # integer N means "best N"
    try:
        k = int(m)
        if k < 1:
            raise ValueError
        k = min(k, nposes)
        idx = np.argsort(scores)[:k]
        return [int(i) for i in idx]
    except Exception as e:
        raise ValueError(f"Invalid --trace-pose value: {mode!r} (use best, first, or integer N)") from e


POSE_LAYER_CHOICES = ("scatter", "density", "hexbin", "trace", "centroid")
POSE_LAYER_ORDER_TOP_TO_BOTTOM = ("centroid", "scatter", "trace", "hexbin", "density")


def _normalize_pose_layers(raw_layers: list[str] | None) -> list[str]:
    """Normalize, validate, deduplicate, and order selected pose layers."""
    if raw_layers is None or len(raw_layers) == 0:
        parsed: list[str] = ["density"]
    else:
        parsed = []
        for chunk in raw_layers:
            parts = [p.strip().lower() for p in str(chunk).split(",")]
            parsed.extend([p for p in parts if p])
        if len(parsed) == 0:
            parsed = ["density"]

    invalid = [p for p in parsed if p not in POSE_LAYER_CHOICES]
    if invalid:
        valid_str = ", ".join(POSE_LAYER_CHOICES)
        raise SystemExit(f"Invalid --pose-layer value(s): {', '.join(invalid)}. Valid values: {valid_str}")

    selected = set(parsed)
    return [layer for layer in POSE_LAYER_ORDER_TOP_TO_BOTTOM if layer in selected]


def _infer_map_title(reference_set_id: str, explicit_title: str | None) -> str:
    """Return CLI title override, or derive one from the reference set id."""
    if explicit_title:
        return explicit_title
    return f"dockmap multi-set (ref={reference_set_id})"


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="dockmap",
        description="2D surface map of docking sites + PPI overlay",
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  dockmap --set s1:prot.pdb:poses.pdb:scores.txt --ppi-file ppi.txt\n"
            "  dockmap ... --map hammer --pose-layer centroid --cluster-contour outline\n"
            "  dockmap ... --cluster-contour filled --cluster-contour-color black"
        ),
    )

    # -----------------------
    # Inputs
    # -----------------------
    g_in = ap.add_argument_group("Inputs")
    g_in.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=None,
        help=(
            "Docking set descriptor (repeatable): "
            "set_id:target.pdb:poses.pdb:scores.txt. "
            "Example: --set s1:target1.pdb:poses1.pdb:scores1.txt --set s2:target2.pdb:poses2.pdb:scores2.txt"
        ),
    )
    g_in.add_argument(
        "--ppi-file",
        required=True,
        help="Interface residue list file (one residue per line like: 'Chain  X  RES  123').",
    )
    g_in.add_argument(
        "--align-protein",
        default="ca_rigid",
        choices=["ca_rigid"],
        help="Protein alignment mode. 'ca_rigid' = C-alpha rigid alignment to reference set.",
    )
    g_in.add_argument(
        "--reference-set",
        default=None,
        help="Reference set_id for protein alignment. Default: first provided --set.",
    )
    g_in.add_argument(
        "--no-align",
        action="store_true",
        help="Disable default alignment step and use raw coordinates as-is.",
    )

    # -----------------------
    # Surface
    # -----------------------
    g_surf = ap.add_argument_group("Surface")
    g_surf.add_argument(
        "--radius-scale",
        type=float,
        default=1.4,
        help=(
            "QuickSurf atomic radius multiplier used to build the density field. "
            "Higher values produce a smoother, more 'inflated' surface (fewer fine details); "
            "lower values follow the atomic envelope more tightly."
        ),
    )
    g_surf.add_argument(
        "--density-isovalue",
        type=float,
        default=1.0,
        help=(
            "Iso-value of the density field used to extract the surface. "
            "Higher values generally produce a tighter surface; lower values produce a looser surface. "
            "If the surface looks too 'puffy' or too 'tight', adjust this together with --radius-scale."
        ),
    )
    g_surf.add_argument(
        "--grid-spacing",
        type=float,
        default=1.0,
        help=(
            "Grid spacing (Å) for sampling the density field before extracting the surface. "
            "Smaller values give smoother, higher-resolution meshes but increase time/memory; "
            "larger values are faster but can look blocky/rugged."
        ),
    )
    g_surf.add_argument(
        "--surface-quality",
        default="max",
        choices=["max", "med", "low"],
        help=(
            "QuickSurf quality preset controlling internal kernel cutoff / sampling effort. "
            "Choices: "
            "'max' = highest quality (slower, most detailed); "
            "'med' = balanced; "
            "'low' = fastest and smoothest/least detailed."
        ),
    )
    g_surf.add_argument(
        "--cache-surface",
        default=None,
        help=(
            "Cache the computed surface mesh to/from a .npz file to speed up repeated runs. "
            "Use a different cache filename when changing surface parameters "
            "(--radius-scale/--density-isovalue/--grid-spacing/--surface-quality) to avoid reusing the wrong mesh."
        ),
    )

    # -----------------------
    # PPI
    # -----------------------
    g_ppi = ap.add_argument_group("PPI")
    g_ppi.add_argument(
        "--ppi-footprint",
        action="append",
        choices=["residue_points", "atom_contour"],
        default=None,
        help=(
            "How to display the protein–protein interface (PPI) region on the 2D map. "
            "You can provide this option multiple times to enable multiple overlays. "
            "If omitted, defaults to 'atom_contour'.\n"
            "Examples:\n"
            "  --ppi-footprint residue_points\n"
            "      one point per interface residue (fast, clean, good overview)\n"
            "  --ppi-footprint atom_contour\n"
            "      atom-cloud contour/outline from interface residues (more detailed; can be slower)\n"
            "  --ppi-footprint residue_points --ppi-footprint atom_contour\n"
            "      draw both overlays"
        ),
    )
    g_ppi.add_argument(
        "--ppi-residue-point",
        default="sc_com",
        choices=["ca", "res_com", "sc_com"],
        help=(
            "When --ppi-footprint residue_points, choose the representative point for each residue. "
            "Choices: "
            "'ca' = Cα atom (or residue mean if missing); "
            "'res_com' = center of geometry of all atoms in the residue; "
            "'sc_com' = center of geometry of side-chain atoms only (falls back to residue mean if no side-chain atoms)."
        ),
    )
    g_ppi.add_argument(
        "--ppi-atom-filter",
        default="near_surface",
        choices=["all_heavy", "near_surface", "sasa"],
        help=(
            "When --ppi-footprint atom_contour, choose which atoms from the interface residues contribute to the footprint. "
            "Choices: "
            "'all_heavy' = all non-hydrogen atoms (always shows a footprint, even for very smooth/coarse surfaces); "
            "'near_surface' = keep only atoms within --ppi-near-surface-eps Å of the extracted surface "
            "(best when interface is surface-exposed); "
            "'sasa' = solvent-accessible atoms only (reserved/not implemented unless enabled)."
        ),
    )
    g_ppi.add_argument(
        "--ppi-near-surface-eps",
        type=float,
        default=1.5,
        help=(
            "Distance cutoff (Å) used by --ppi-atom-filter near_surface. "
            "Larger values keep more atoms and are recommended for very smooth/coarse QuickSurf meshes "
            "(e.g., grid-spacing 1.5 and large radius-scale)."
        ),
    )

    # -----------------------
    # Projection
    # -----------------------
    g_proj = ap.add_argument_group("Projection")
    g_proj.add_argument(
        "--map",
        default="mollweide",
        choices=["equirect", "mollweide", "hammer"],
        help=(
            "2D spherical map projection used to flatten (theta, phi) onto the figure. "
            "Choices: "
            "'equirect' = simple lon/lat rectangle (fast, distorted near poles); "
            "'mollweide' = equal-area world map (good default for densities); "
            "'hammer' = equal-area, slightly different shape/distortion tradeoff."
        ),
    )
    g_proj.add_argument(
        "--pose-projection",
        default="nearest",
        choices=["nearest", "raycast"],
        help=(
            "How each peptide center is mapped to the protein surface before converting to (theta, phi). "
            "Choices: "
            "'nearest' = closest point on the surface mesh (robust default); "
            "'raycast' = cast a ray from the protein center through the peptide center and take the first surface hit "
            "(can be better for deep pockets but may miss if geometry is complex)."
        ),
    )
    g_proj.add_argument(
        "--peptide-center",
        default="com",
        choices=["com", "cog"],
        help=(
            "How the peptide pose is reduced to a single representative point. "
            "Choices: "
            "'com' = center of mass (uses element masses; default); "
            "'cog' = center of geometry (simple coordinate mean; ignores masses)."
        ),
    )
    g_proj.add_argument(
        "--seam-rotate",
        default="auto",
        help=(
            "Rotate the map seam (longitude origin). "
            "Use 'auto' to place the seam away from the main pose cluster, or provide a number (degrees) "
            "to rotate longitude by that amount before 2D projection. "
            "Use 0 for no rotation. Example: --seam-rotate 90."
        ),
    )
    g_proj.add_argument(
        "--cluster-distance",
        type=float,
        default=15.0,
        help=(
            "Angular distance threshold (degrees) used to cluster poses on the sphere by (theta, phi). "
            "Poses are connected when great-circle distance is <= this threshold; singleton poses become "
            "single-pose clusters."
        ),
    )

    # -----------------------
    # Output
    # -----------------------
    g_out = ap.add_argument_group("Output")
    g_out.add_argument("--out-prefix", default="dockmap", help="Output file prefix.")
    g_out.add_argument(
        "--map-title",
        default=None,
        help="Optional 2D map title override. If omitted, uses '<protein> vs <peptides>' from input filenames.",
    )
    g_out.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Figure format for the 2D map.")
    g_out.add_argument("--write-csv", action="store_true", default=True, help="Write CSV outputs (poses and PPI UV).")

    g_out.add_argument("--export-mesh", action="store_true", default=False, help="Export the QuickSurf surface mesh.")
    g_out.add_argument("--mesh-format", default="ply", choices=["obj", "ply", "stl"], help="Mesh export format.")
    g_out.add_argument("--mesh-path", default=None, help="Explicit mesh output path (optional).")
    g_out.add_argument(
        "--export-aligned-pdbs",
        action="store_true",
        default=False,
        help=(
            "Write transformed/aligned coordinates for each set as PDB files "
            "(protein + multi-model poses)."
        ),
    )
    g_out.add_argument(
        "--aligned-pdb-dir",
        default=None,
        help=(
            "Output directory for --export-aligned-pdbs files. "
            "Default: '<out-prefix>_aligned_pdbs'."
        ),
    )
    g_out.add_argument(
        "--mesh-vertex-scalar",
        default="none",
        choices=["none", "density", "curv_proxy"],
        help=(
            "If exporting PLY, include a per-vertex scalar for coloring. "
            "'density' uses the sampled QuickSurf field; 'curv_proxy' uses a curvature-like Laplacian magnitude."
        ),
    )
    g_out.add_argument("--mesh-scalar-name", default=None, help="Name of scalar property in PLY (default auto).")

    # -----------------------
    # Advanced
    # -----------------------
    g_adv = ap.add_argument_group("Advanced")
    g_adv.add_argument(
        "--pose-layer",
        action="append",
        default=None,
        help=(
            "How peptide poses are drawn on the 2D map. "
            "You can pass this flag multiple times (or comma-separate values) to draw multiple layers. "
            "Layers are composited in fixed top-to-bottom order: centroid, scatter, trace, hexbin, density. "
            "Choices: "
            "'scatter' = plot one marker per pose (best for small N or top-N subsets); "
            "'density' = smooth heatmap on a regular lon/lat grid (good default for many poses); "
            "'hexbin' = hexagonal bin counts (crisper binned view, less smoothing than density); "
            "'trace' = draw peptide backbone trace (Cα atoms + connecting line) for selected pose(s); "
            "'centroid' = one marker per cluster centroid, labeled as 'rank:size\n<cluster_avg_vina>' (example: 1:215\n<-8.34>).\n"
            "Examples:\n"
            "  --pose-layer density\n"
            "  --pose-layer centroid --pose-layer scatter\n"
            "  --pose-layer centroid,scatter,trace"
        ),
    )
    g_adv.add_argument(
        "--cluster-contour",
        default="none",
        choices=["none", "outline", "filled"],
        help=(
            "Draw per-cluster smooth KDE/isodensity contours in projected 2D map space. "
            "'outline' draws contour lines; 'filled' adds translucent fills + outlines."
        ),
    )
    g_adv.add_argument(
        "--cluster-contour-color",
        default=None,
        help=(
            "Optional single matplotlib color for all cluster contours (e.g., 'black', '#3366ff'). "
            "If omitted, contours are colored by cluster size rank from red (largest) to blue (smallest)."
        ),
    )
    g_adv.add_argument(
        "--trace-pose",
        default="best",
        help=(
            "When --pose-layer trace, choose which pose(s) to trace. "
            "Values: "
            "'best' = trace the best-scoring pose (lowest Vina score); "
            "'first' = trace the first pose in the peptides file; "
            "'N' (integer) = trace the best N poses by score (e.g., 5 traces the 5 best poses)."
        ),
    )
    g_adv.add_argument(
        "--weight",
        default="exp",
        choices=["none", "exp", "linear"],
        help=(
            "How poses are weighted when aggregating into 'density' or 'hexbin' layers (ignored for scatter/trace/centroid). "
            "Choices: "
            "'none' = all poses contribute equally; "
            "'linear' = linearly rescale weights by score (emphasizes better scores); "
            "'exp' = exponential weight from Vina score (strongly emphasizes best scores)."
        ),
    )
    g_adv.add_argument(
        "--pose-density-sigma",
        type=float,
        default=2.0,
        help=(
            "Gaussian smoothing width (in density-grid pixels) for --pose-layer density. "
            "Smaller values make broader/smoother density blobs; "
            "larger values make the density tighter/sharper around cluster members."
        ),
    )
    g_adv.add_argument(
        "--background",
        default="none",
        choices=["none", "curvature", "radial"],
        help=(
            "Optional background shading derived from the surface mesh to help orient the map. "
            "Choices: "
            "'none' = no background; "
            "'curvature' = curvature-like proxy from mesh Laplacian magnitude (highlights ridges/valleys); "
            "'radial' = distance from protein center (highlights bulges/indentations; may look flat on near-spherical meshes)."
        ),
    )
    g_adv.add_argument(
        "--background-smooth",
        type=int,
        default=0,
        help=(
            "Number of neighbor-averaging iterations applied to the background scalar on the mesh before plotting. "
            "Use 0 for no smoothing; small values (e.g., 2–5) reduce noise and produce a cleaner relief."
        ),
    )



    # Background colorbar controls (NEW)
    g_adv.add_argument(
        "--background-colorbar",
        action="store_true",
        default=False,
        help="Show a colorbar for the background shading (only when --background != none).",
    )
    g_adv.add_argument(
        "--background-colorbar-location",
        default="right",
        choices=["right", "bottom"],
        help="Colorbar placement.",
    )
    g_adv.add_argument(
        "--background-colorbar-mode",
        default="norm",
        choices=["norm", "raw"],
        help=(
            "Colorbar scale mode. "
            "'norm' = show normalized [0..1] (default). "
            "'raw' = show the plotted scalar in its native units (use --background-colorbar-vmin/vmax to control range)."
        ),
    )
    g_adv.add_argument(
        "--background-colorbar-label",
        default=None,
        help="Override the colorbar label text (default depends on --background and mode).",
    )
    g_adv.add_argument(
        "--background-colorbar-vmin",
        type=float,
        default=None,
        help="Lower bound for raw-mode colorbar scaling (only used if --background-colorbar-mode raw).",
    )
    g_adv.add_argument(
        "--background-colorbar-vmax",
        type=float,
        default=None,
        help="Upper bound for raw-mode colorbar scaling (only used if --background-colorbar-mode raw).",
    )




    g_adv.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity (-v=INFO, -vv=DEBUG).")

    # Pose label controls (existing behavior kept: mainly useful for scatter/trace)
    g_adv.add_argument(
        "--pose-label",
        default="none",
        choices=["none", "first", "best", "topN"],
        help=(
            "Pose label mode. Labels are drawn as 'LIG####'. "
            "Choices: none, first, best (lowest score), or topN (best N by score)."
        ),
    )
    g_adv.add_argument(
        "--pose-label-top",
        dest="pose_label_top",
        type=int,
        default=5,
        help="If --pose-label topN, how many best poses to label.",
    )

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)
    args.pose_layer = _normalize_pose_layers(args.pose_layer)
    configure_logging(args.verbose)

    if args.cluster_distance <= 0:
        raise SystemExit("--cluster-distance must be > 0 degrees.")
    if args.pose_density_sigma <= 0:
        raise SystemExit("--pose-density-sigma must be > 0.")

    log.info("dockmap pipeline start")
    log.debug("Arguments: %s", vars(args))

    if not args.sets:
        raise SystemExit("At least one --set is required.")

    log.info("Input options | n_sets=%d | ppi_file=%s", len(args.sets), args.ppi_file)
    log.info(
        "Surface options | radius_scale=%s | density_isovalue=%s | grid_spacing=%s | surface_quality=%s | cache_surface=%s",
        args.radius_scale,
        args.density_isovalue,
        args.grid_spacing,
        args.surface_quality,
        args.cache_surface,
    )
    log.info(
        "Projection options | map=%s | pose_layers=%s | pose_projection=%s | peptide_center=%s | seam_rotate=%s | cluster_distance_deg=%s",
        args.map,
        ",".join(args.pose_layer),
        args.pose_projection,
        args.peptide_center,
        args.seam_rotate,
        args.cluster_distance,
    )
    log.info(
        "PPI options | ppi_footprint=%s | ppi_atom_filter=%s | ppi_near_surface_eps=%s | ppi_residue_point=%s",
        ",".join(args.ppi_footprint) if args.ppi_footprint else "atom_contour",
        args.ppi_atom_filter,
        args.ppi_near_surface_eps,
        args.ppi_residue_point,
    )
    log.info(
        "Output options | out_prefix=%s | format=%s | write_csv=%s | export_mesh=%s | mesh_format=%s | mesh_path=%s | export_aligned_pdbs=%s | aligned_pdb_dir=%s",
        args.out_prefix,
        args.format,
        args.write_csv,
        args.export_mesh,
        args.mesh_format,
        args.mesh_path,
        args.export_aligned_pdbs,
        args.aligned_pdb_dir,
    )

    # ---- Load multi-set inputs
    with Timer("Load all docking sets", log):
        dock_sets = _load_dock_sets(args.sets)
    total_poses = sum(len(s.poses) for s in dock_sets)
    log.info("Loaded %d sets | total poses=%d", len(dock_sets), total_poses)

    reference_set_id = args.reference_set or dock_sets[0].set_id
    set_by_id = {s.set_id: s for s in dock_sets}
    if reference_set_id not in set_by_id:
        raise SystemExit(f"--reference-set {reference_set_id!r} not found in provided --set values.")
    reference_set = set_by_id[reference_set_id]
    protein_atoms = reference_set.protein_atoms

    # ---- Optional target alignment (default on, but skipped for single-set or --no-align)
    aligned_sets: list[DockSet] = []
    alignment_rows: list[dict[str, object]] = []
    perform_alignment = (not args.no_align) and (len(dock_sets) > 1)
    ref_ca: dict[tuple[str, int, str], AtomRecord] = {}
    if perform_alignment:
        ref_ca = _extract_target_ca_by_residue(reference_set.protein_atoms)
        if not ref_ca:
            raise SystemExit(f"Reference set {reference_set_id!r} has no C-alpha atoms.")

    for ds in dock_sets:
        if (not perform_alignment) or ds.set_id == reference_set_id:
            aligned_sets.append(ds)
            alignment_rows.append(
                {
                    "set_id": ds.set_id,
                    "reference_set_id": reference_set_id,
                    "n_ca_matched": len(ref_ca) if perform_alignment else 0,
                    "rmsd_before": 0.0,
                    "rmsd_after": 0.0,
                    "ligand_displacement_min": 0.0,
                    "ligand_displacement_max": 0.0,
                    "ligand_displacement_mean": 0.0,
                    "ligand_displacement_std": 0.0,
                    "alignment_enabled": perform_alignment,
                }
            )
            continue

        mov_ca = _extract_target_ca_by_residue(ds.protein_atoms)
        if set(ref_ca.keys()) != set(mov_ca.keys()):
            only_ref = sorted(set(ref_ca.keys()) - set(mov_ca.keys()))[:8]
            only_mov = sorted(set(mov_ca.keys()) - set(ref_ca.keys()))[:8]
            raise SystemExit(
                f"Set {ds.set_id!r} failed sequence/C-alpha sanity check vs reference {reference_set_id!r}. "
                f"Only in reference (sample): {only_ref}; only in set (sample): {only_mov}"
            )

        keys = sorted(ref_ca.keys(), key=lambda k: (k[0], k[1], k[2]))
        ref_xyz = np.array([ref_ca[k].coord for k in keys], dtype=float)
        mov_xyz = np.array([mov_ca[k].coord for k in keys], dtype=float)
        rmsd_before = _rmsd(mov_xyz, ref_xyz)
        r, t = _kabsch_rigid_transform(mov_xyz, ref_xyz)
        mov_aligned_xyz = mov_xyz @ r + t
        rmsd_after = _rmsd(mov_aligned_xyz, ref_xyz)
        if not np.isfinite(rmsd_after):
            raise SystemExit(f"Alignment failed numerically for set {ds.set_id!r}.")

        aligned_protein = _transform_atoms(ds.protein_atoms, r, t)
        aligned_poses = []
        pose_displacements = []
        for pose in ds.poses:
            before_coords = coords_from_atoms(pose.peptide_atoms)
            before_center = before_coords.mean(axis=0)
            pa = _transform_atoms(pose.peptide_atoms, r, t)
            after_coords = coords_from_atoms(pa)
            after_center = after_coords.mean(axis=0)
            pose_displacements.append(float(np.linalg.norm(after_center - before_center)))
            aligned_poses.append(Pose(pose_id=pose.pose_id, peptide_atoms=pa, vina_score=pose.vina_score))

        pd = np.array(pose_displacements, dtype=float)
        aligned_sets.append(
            DockSet(
                set_id=ds.set_id,
                protein_path=ds.protein_path,
                peptides_path=ds.peptides_path,
                scores_path=ds.scores_path,
                protein_atoms=aligned_protein,
                poses=aligned_poses,
            )
        )
        alignment_rows.append(
            {
                "set_id": ds.set_id,
                "reference_set_id": reference_set_id,
                "n_ca_matched": int(len(keys)),
                "rmsd_before": float(rmsd_before),
                "rmsd_after": float(rmsd_after),
                "ligand_displacement_min": float(np.min(pd)) if pd.size else 0.0,
                "ligand_displacement_max": float(np.max(pd)) if pd.size else 0.0,
                "ligand_displacement_mean": float(np.mean(pd)) if pd.size else 0.0,
                "ligand_displacement_std": float(np.std(pd)) if pd.size else 0.0,
                "alignment_enabled": True,
            }
        )

    # keep plotting frame as reference set (aligned if requested)
    aligned_ref = next(s for s in aligned_sets if s.set_id == reference_set_id)
    protein_atoms = aligned_ref.protein_atoms
    poses = [p for s in aligned_sets for p in s.poses]
    pose_set_ids = [s.set_id for s in aligned_sets for _ in s.poses]
    log.info("Alignment summary (reference=%s, enabled=%s):", reference_set_id, perform_alignment)
    for row in alignment_rows:
        log.info(
            "  set=%s | n_ca=%s | rmsd_before=%.3f | rmsd_after=%.3f | lig_disp_mean=%.3f",
            row["set_id"],
            row["n_ca_matched"],
            float(row["rmsd_before"]),
            float(row["rmsd_after"]),
            float(row["ligand_displacement_mean"]),
        )

    if args.export_aligned_pdbs:
        out_prefix = Path(args.out_prefix)
        aligned_dir = Path(args.aligned_pdb_dir) if args.aligned_pdb_dir else out_prefix.with_name(out_prefix.name + "_aligned_pdbs")
        aligned_dir.mkdir(parents=True, exist_ok=True)
        for ds in aligned_sets:
            protein_path = aligned_dir / f"{ds.set_id}_protein_aligned.pdb"
            poses_path = aligned_dir / f"{ds.set_id}_poses_aligned.pdb"
            write_pdb_atoms(protein_path, ds.protein_atoms, record="ATOM")
            write_pdb_poses(poses_path, ds.poses, record="HETATM")
        log.info("Exported aligned PDBs for %d sets to: %s", len(aligned_sets), aligned_dir)

    with Timer("Parse PPI residue list", log):
        ppi = parse_ppi_file(args.ppi_file)
    log.info("Loaded %d PPI residues", len(ppi))

    # ---- Validate PPI residues exist in protein (fail-fast)
    prot_residues, chain_counts = protein_residue_inventory(protein_atoms)
    ok, rep = validate_ppi_residues_exist(ppi, prot_residues, max_examples=25)

    log.info(
        "Protein chains present: %s",
        ", ".join(f"{c}({chain_counts.get(c,0)} atoms)" for c in rep["protein_chains"]),
    )
    log.info("PPI chains in contacts file: %s", ", ".join(rep["ppi_chains"]) if rep["ppi_chains"] else "(none)")
    log.info(
        "PPI residues: %d  | matched: %d  | missing: %d",
        rep["ppi_total"],
        rep["present_count"],
        rep["missing_count"],
    )

    if not ok:
        prot_chain_set = set(rep["protein_chains"])
        ppi_chain_set = set(rep["ppi_chains"])
        missing_chains = sorted(ppi_chain_set - prot_chain_set)
        if missing_chains:
            log.error(
                "Chain mismatch: contacts file contains chains not present in protein: %s",
                ", ".join(missing_chains),
            )

        if rep["missing_examples"]:
            ex = rep["missing_examples"]
            ex_str = ", ".join([f"{r.chain}:{r.resseq}{r.icode or ''}" for r in ex])
            log.error("Example missing residues (first %d): %s", len(ex), ex_str)

        log.error(
            "PPI validation failed: contacts residues do not match protein residue IDs. "
            "Fix chain IDs and/or residue numbering (renumber, or regenerate contacts from this PDB)."
        )
        raise SystemExit(2)

    prot_coords = coords_from_atoms(protein_atoms)
    protein_center = center_of_geometry(prot_coords)

    # ---- Surface mesh (and optionally density field)
    cache_path = Path(args.cache_surface) if args.cache_surface else None
    surf_spec = QuickSurfSpec(
        radius_scale=args.radius_scale,
        density_isovalue=args.density_isovalue,
        grid_spacing=args.grid_spacing,
        surface_quality=args.surface_quality,
        cache_path=cache_path,
    )

    need_field = bool(args.export_mesh and args.mesh_vertex_scalar == "density")
    if need_field:
        with Timer("Build QuickSurf mesh + density field (for vertex density export)", log):
            mesh, field = build_quicksurf_mesh(protein_atoms, surf_spec, return_field=True)
    else:
        with Timer("Build QuickSurf mesh", log):
            mesh = build_quicksurf_mesh(protein_atoms, surf_spec)

    log.info("Surface mesh: %d vertices, %d faces", mesh.vertices.shape[0], mesh.faces.shape[0])

    # ---- Map poses (centers)
    with Timer("Project peptide centers to surface + map to spherical UV", log):
        pose_theta, pose_phi, pose_dist, pose_ids, scores, mapped_set_ids = [], [], [], [], [], []
        for i_pose, pose in enumerate(poses):
            pep_coords = coords_from_atoms(pose.peptide_atoms)
            if pep_coords.size == 0:
                continue

            if args.peptide_center == "cog":
                q = pep_coords.mean(axis=0)
            else:
                elems = [a.element for a in pose.peptide_atoms]
                q = center_of_mass(pep_coords, elems)

            if args.pose_projection == "raycast":
                pr = project_point_to_surface_raycast(q, protein_center, mesh)
            else:
                pr = project_point_to_surface_nearest(q, mesh)

            th, ph = surface_point_to_spherical_uv(pr.point, protein_center)
            pose_theta.append(th)
            pose_phi.append(ph)
            pose_dist.append(pr.distance)
            sid = pose_set_ids[i_pose] if i_pose < len(pose_set_ids) else "set000"
            pose_ids.append(f"{sid}:{pose.pose_id}")
            mapped_set_ids.append(sid)
            scores.append(pose.vina_score)

        pose_theta = np.array(pose_theta, float)
        pose_phi = np.array(pose_phi, float)
        pose_dist = np.array(pose_dist, float)
        scores = np.array(scores, float)

    log.info("Mapped poses: %d", len(pose_theta))

    # ---- Seam rotation
    if args.seam_rotate == "auto":
        rot = auto_seam_rotation(pose_theta, weights=None)
        log.info("Seam rotation: auto -> %.2f deg", np.rad2deg(rot))
    else:
        rot = np.deg2rad(float(args.seam_rotate))
        log.info("Seam rotation: user -> %.2f deg", float(args.seam_rotate))

    # Apply seam rotation to pose longitudes
    pose_theta = apply_seam_rotation(pose_theta, rot)

    # ---- PPI overlay mapping (supports one or BOTH overlays)
    ppi_contour_theta = ppi_contour_phi = None
    ppi_points_theta = ppi_points_phi = None
    ppi_points_labels = None

    ppi_modes = args.ppi_footprint or ["atom_contour"]
    seen = set()
    ppi_modes = [m for m in ppi_modes if not (m in seen or seen.add(m))]

    log.info(
        "Map PPI footprint step options | modes=%s | atom_filter=%s | near_surface_eps=%s | residue_point_mode=%s",
        ",".join(ppi_modes),
        args.ppi_atom_filter,
        args.ppi_near_surface_eps,
        args.ppi_residue_point,
    )

    with Timer("Map PPI footprint to UV", log):
        if "atom_contour" in ppi_modes:
            th, ph = ppi_atom_cloud_uv(
                protein_atoms,
                ppi,
                mesh,
                protein_center,
                atom_filter=args.ppi_atom_filter,
                near_surface_eps=args.ppi_near_surface_eps,
            )
            if len(th) > 0:
                th = apply_seam_rotation(th, rot)
                ppi_contour_theta, ppi_contour_phi = th, ph

        if "residue_points" in ppi_modes:
            th, ph, labs = ppi_residue_points_uv(
                protein_atoms,
                ppi,
                mesh,
                protein_center,
                residue_point_mode=args.ppi_residue_point,
            )
            if len(th) > 0:
                th = apply_seam_rotation(th, rot)
                ppi_points_theta, ppi_points_phi = th, ph
                ppi_points_labels = labs
            else:
                ppi_points_labels = None

    n_cont = 0 if ppi_contour_theta is None else len(ppi_contour_theta)
    n_pts = 0 if ppi_points_theta is None else len(ppi_points_theta)
    log.info("Mapped PPI contour points: %d", n_cont)
    log.info("Mapped PPI residue points: %d", n_pts)

    # ---- Background layer preparation (mesh vertices -> UV + scalar)
    mesh_theta = mesh_phi = mesh_scalar = mesh_scalar_raw = None

    if args.background != "none":
        log.info(
            "Background step options | background=%s | background_smooth=%s | colorbar=%s | colorbar_mode=%s | colorbar_location=%s | colorbar_label=%s | vmin=%s | vmax=%s",
            args.background,
            args.background_smooth,
            args.background_colorbar,
            args.background_colorbar_mode,
            args.background_colorbar_location,
            args.background_colorbar_label,
            args.background_colorbar_vmin,
            args.background_colorbar_vmax,
        )
        with Timer(f"Compute background scalar ({args.background})", log):
            vtx = mesh.vertices
            th = np.empty((vtx.shape[0],), float)
            ph = np.empty((vtx.shape[0],), float)
            for i in range(vtx.shape[0]):
                th[i], ph[i] = surface_point_to_spherical_uv(vtx[i], protein_center)
            th = apply_seam_rotation(th, rot)

            if args.background == "radial":
                sc_bg = radial_distance(mesh, protein_center)
            else:
                sc_bg = vertex_laplacian_magnitude(mesh)

            if args.background_smooth > 0:
                log.info("Background smoothing: %d iterations", args.background_smooth)
                sc_bg = smooth_scalar_neighbor_average(mesh, sc_bg, n_iter=args.background_smooth)

            mesh_scalar_raw = sc_bg.copy()
            sc_bg = normalize_01(sc_bg)
            mesh_theta, mesh_phi, mesh_scalar = th, ph, sc_bg

    # ---- Export mesh (optional)
    if args.export_mesh:
        log.info(
            "Mesh export step options | mesh_format=%s | mesh_path=%s | mesh_vertex_scalar=%s | mesh_scalar_name=%s",
            args.mesh_format,
            args.mesh_path,
            args.mesh_vertex_scalar,
            args.mesh_scalar_name,
        )
        out_prefix = Path(args.out_prefix)
        if args.mesh_path:
            mesh_path = Path(args.mesh_path)
        else:
            mesh_path = out_prefix.with_name(out_prefix.name + "_quicksurf").with_suffix("." + args.mesh_format)

        vscalar = None
        sname = args.mesh_scalar_name
        if args.mesh_vertex_scalar != "none":
            if args.mesh_format != "ply":
                raise ValueError("Per-vertex scalar export is only supported for PLY.")
            if args.mesh_vertex_scalar == "density":
                sname = sname or "density"
                vscalar = sample_field_trilinear(field, mesh.vertices)  # type: ignore[name-defined]
            else:
                sname = sname or "curv"
                vscalar = vertex_laplacian_magnitude(mesh)

            vscalar = normalize_01(vscalar)

        with Timer(f"Export mesh ({args.mesh_format})", log):
            write_mesh(mesh, mesh_path, args.mesh_format, vertex_scalar=vscalar, scalar_name=sname or "scalar")
        log.info("Wrote mesh: %s", mesh_path)

    # ---- Clusters from mapped pose centers (used for contours, centroid layer, and CSV outputs)
    log.info(
        "Clustering step options | cluster_distance_deg=%s | cluster_contour=%s | cluster_contour_color=%s",
        args.cluster_distance,
        args.cluster_contour,
        args.cluster_contour_color,
    )
    cluster_threshold_rad = np.deg2rad(float(args.cluster_distance))
    raw_cluster_labels = cluster_connected_components(pose_theta, pose_phi, cluster_threshold_rad)
    ordered_idx, cluster_ids, cluster_summaries = reorder_clusters_and_poses(
        pose_theta,
        pose_phi,
        scores,
        pose_ids,
        raw_cluster_labels,
    )

    cluster_theta = np.array([float(row["theta_centroid"]) for row in cluster_summaries], dtype=float)
    cluster_phi = np.array([float(row["phi_centroid"]) for row in cluster_summaries], dtype=float)
    cluster_avg_vina_scores = np.array([float(row["vina_score_avg"]) for row in cluster_summaries], dtype=float)
    cluster_p_values = np.array([float(row["p_value"]) for row in cluster_summaries], dtype=float)

    # ---- Pose labels selection (scatter/trace only)
    pose_labels: list[str] | None = None
    if ({"scatter", "trace"} & set(args.pose_layer)) and args.pose_label != "none":
        n = len(pose_ids)
        labels = [""] * n

        if args.pose_label == "first":
            if n > 0:
                labels[0] = _pose_id_to_ligid(pose_ids[0])
        elif args.pose_label == "best":
            if n > 0:
                ib = int(np.argmin(scores))
                labels[ib] = _pose_id_to_ligid(pose_ids[ib])
        elif args.pose_label == "topN":
            if n > 0:
                topn = max(1, int(args.pose_label_top))
                idx = np.argsort(scores)[: min(topn, n)]
                for i in idx:
                    labels[int(i)] = _pose_id_to_ligid(pose_ids[int(i)])
        pose_labels = labels

    # ---- Trace poses (only when pose-layer trace)
    trace_lines: list[tuple[np.ndarray, np.ndarray]] | None = None
    trace_labels: list[str] | None = None

    if "trace" in args.pose_layer:
        log.info("Trace step options | trace_pose=%s | pose_label=%s | pose_label_top=%s", args.trace_pose, args.pose_label, args.pose_label_top)
        nposes = len(poses)
        if nposes == 0:
            raise SystemExit("No poses available for trace.")

        trace_indices = _select_pose_indices_for_trace(args.trace_pose, scores, nposes)
        if len(trace_indices) == 0:
            raise SystemExit("No poses selected for trace.")

        trace_lines = []
        trace_labels = []

        with Timer("Project trace CA atoms to surface + map to spherical UV", log):
            for trace_idx in trace_indices:
                tr_pose = poses[int(trace_idx)]
                ca_atoms = _extract_ca_trace_atoms(tr_pose.peptide_atoms)
                if len(ca_atoms) == 0:
                    raise SystemExit(f"Trace requested but no CA atoms found in pose {tr_pose.pose_id}.")

                ths, phs = [], []
                for a in ca_atoms:
                    pr = project_point_to_surface_nearest(a.coord, mesh)
                    th, ph = surface_point_to_spherical_uv(pr.point, protein_center)
                    ths.append(th)
                    phs.append(ph)

                tth = apply_seam_rotation(np.array(ths, float), rot)
                tph = np.array(phs, float)
                trace_lines.append((tth, tph))
                trace_labels.append(_pose_id_to_ligid(tr_pose.pose_id))

        # In trace mode, if pose_labels are requested, show labels for traced poses (not all pose centers)
        if args.pose_label != "none":
            pose_labels = trace_labels[:]  # matches viz.py behavior: label first CA of each trace

    # ---- Plot
    out_prefix = Path(args.out_prefix)
    fig_path = out_prefix.with_suffix("." + args.format)

    ps = PlotSpec(
        map_name=args.map,
        map_title=_infer_map_title(reference_set_id, args.map_title),
        pose_layers=tuple(args.pose_layer),
        weight_mode=args.weight,
        out_format=args.format,
        background=args.background,
        pose_density_sigma=args.pose_density_sigma,
    )



    # Choose which scalar to send to viz for background:
    # - viz can plot normalized or raw; it will handle scaling, but it needs the raw field for raw mode.
    mesh_scalar_for_plot = mesh_scalar_raw if (args.background_colorbar_mode == "raw") else mesh_scalar



    log.info(
        "Render step options | map=%s | pose_layers=%s | weight=%s | pose_density_sigma=%s | format=%s | background=%s | cluster_contour=%s",
        args.map,
        ",".join(args.pose_layer),
        args.weight,
        args.pose_density_sigma,
        args.format,
        args.background,
        args.cluster_contour,
    )

    with Timer("Render 2D map", log):
        plot_map(
            pose_theta=pose_theta,
            pose_phi=pose_phi,
            pose_scores=scores,
            out_path=fig_path,
            plot_spec=ps,
            ppi_contour_theta=ppi_contour_theta,
            ppi_contour_phi=ppi_contour_phi,
            ppi_points_theta=ppi_points_theta,
            ppi_points_phi=ppi_points_phi,
            ppi_points_labels=ppi_points_labels,
            mesh_theta=mesh_theta,
            mesh_phi=mesh_phi,
            mesh_scalar=mesh_scalar_for_plot,
            pose_labels=pose_labels,
            # UPDATED: pass multiple traces
            trace_lines=trace_lines,
            trace_labels=trace_labels,
            cluster_ids=cluster_ids,
            cluster_theta=cluster_theta,
            cluster_phi=cluster_phi,
            cluster_avg_vina_scores=cluster_avg_vina_scores,
            cluster_p_values=cluster_p_values,
            cluster_contour=args.cluster_contour,
            cluster_contour_color=args.cluster_contour_color,
            background_colorbar=args.background_colorbar,
            background_colorbar_location=args.background_colorbar_location,
            background_colorbar_mode=args.background_colorbar_mode,
            background_colorbar_label=args.background_colorbar_label,
            background_colorbar_vmin=args.background_colorbar_vmin,
            background_colorbar_vmax=args.background_colorbar_vmax,

        )

    log.info("Wrote map: %s", fig_path)

    # ---- CSV outputs
    if args.write_csv:
        log.info("CSV step options | write_csv=%s", args.write_csv)
        with Timer("Write CSV outputs", log):
            pose_csv = out_prefix.with_name(out_prefix.name + "_poses_mapped.csv")
            with pose_csv.open("w", newline="") as f:
                wcsv = csv.writer(f)
                wcsv.writerow([
                    "set_id",
                    "reference_set_id",
                    "pose_id",
                    "vina_score",
                    "theta",
                    "phi",
                    "proj_distance",
                    "cluster_id",
                ])
                for i in ordered_idx:
                    wcsv.writerow(
                        [
                            mapped_set_ids[i],
                            reference_set_id,
                            pose_ids[i],
                            _format_csv_cell(scores[i]),
                            _format_csv_cell(pose_theta[i]),
                            _format_csv_cell(pose_phi[i]),
                            _format_csv_cell(pose_dist[i]),
                            cluster_ids[i],
                        ]
                    )
            log.info("Wrote CSV: %s", pose_csv)

            clusters_csv = out_prefix.with_name(out_prefix.name + "_clusters.csv")
            with clusters_csv.open("w", newline="") as f:
                wcsv = csv.writer(f)
                wcsv.writerow(
                    [
                        "cluster_id",
                        "n_poses",
                        "best_pose_id",
                        "best_vina_score",
                        "vina_score_min",
                        "vina_score_max",
                        "vina_score_avg",
                        "vina_score_stddev",
                        "p_value",
                        "theta_centroid",
                        "phi_centroid",
                    ]
                )
                for row in cluster_summaries:
                    wcsv.writerow(
                        [
                            row["cluster_id"],
                            row["n_poses"],
                            row["best_pose_id"],
                            _format_csv_cell(row["best_vina_score"]),
                            _format_csv_cell(row["vina_score_min"]),
                            _format_csv_cell(row["vina_score_max"]),
                            _format_csv_cell(row["vina_score_avg"]),
                            _format_csv_cell(row["vina_score_stddev"]),
                            _format_csv_cell(row["p_value"]),
                            _format_csv_cell(row["theta_centroid"]),
                            _format_csv_cell(row["phi_centroid"]),
                        ]
                    )
            log.info("Wrote CSV: %s", clusters_csv)

            align_csv = out_prefix.with_name(out_prefix.name + "_alignment_report.csv")
            with align_csv.open("w", newline="") as f:
                wcsv = csv.writer(f)
                wcsv.writerow(
                    [
                        "set_id",
                        "reference_set_id",
                        "n_ca_matched",
                        "rmsd_before",
                        "rmsd_after",
                        "ligand_displacement_min",
                        "ligand_displacement_max",
                        "ligand_displacement_mean",
                        "ligand_displacement_std",
                        "alignment_enabled",
                        "alignment_atom_selection",
                        "alignment_mode",
                        "dockmap_version",
                    ]
                )
                for row in alignment_rows:
                    wcsv.writerow(
                        [
                            row["set_id"],
                            row["reference_set_id"],
                            row["n_ca_matched"],
                            _format_csv_cell(row["rmsd_before"]),
                            _format_csv_cell(row["rmsd_after"]),
                            _format_csv_cell(row["ligand_displacement_min"]),
                            _format_csv_cell(row["ligand_displacement_max"]),
                            _format_csv_cell(row["ligand_displacement_mean"]),
                            _format_csv_cell(row["ligand_displacement_std"]),
                            row["alignment_enabled"],
                            "CA",
                            "rigid_kabsch" if perform_alignment else "none",
                            __version__,
                        ]
                    )
            log.info("Wrote CSV: %s", align_csv)

            align_json = out_prefix.with_name(out_prefix.name + "_alignment_report.json")
            payload = {
                "reference_set_id": reference_set_id,
                "alignment_enabled": perform_alignment,
                "alignment_atom_selection": "CA",
                "alignment_mode": "rigid_kabsch" if perform_alignment else "none",
                "dockmap_version": __version__,
                "sets": alignment_rows,
            }
            align_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            log.info("Wrote JSON: %s", align_json)

            if ppi_contour_theta is not None and ppi_contour_phi is not None:
                ppi_csv = out_prefix.with_name(out_prefix.name + "_ppi_contour_mapped.csv")
                with ppi_csv.open("w", newline="") as f:
                    wcsv = csv.writer(f)
                    wcsv.writerow(["theta", "phi"])
                    for i in range(len(ppi_contour_theta)):
                        wcsv.writerow([
                            _format_csv_cell(ppi_contour_theta[i]),
                            _format_csv_cell(ppi_contour_phi[i]),
                        ])
                log.info("Wrote CSV: %s", ppi_csv)

            if ppi_points_theta is not None and ppi_points_phi is not None:
                ppi_csv2 = out_prefix.with_name(out_prefix.name + "_ppi_residue_points_mapped.csv")
                with ppi_csv2.open("w", newline="") as f:
                    wcsv = csv.writer(f)
                    wcsv.writerow(["theta", "phi", "label"])
                    for i in range(len(ppi_points_theta)):
                        lab = "" if (ppi_points_labels is None or i >= len(ppi_points_labels)) else ppi_points_labels[i]
                        wcsv.writerow([
                            _format_csv_cell(ppi_points_theta[i]),
                            _format_csv_cell(ppi_points_phi[i]),
                            lab,
                        ])
                log.info("Wrote CSV: %s", ppi_csv2)

    log.info("dockmap pipeline complete")
    return 0
