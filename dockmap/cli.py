from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import csv

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
    AtomRecord,
)
from .surface import build_quicksurf_mesh, QuickSurfSpec, sample_field_trilinear
from .project import project_point_to_surface_nearest, project_point_to_surface_raycast
from .mapproj import surface_point_to_spherical_uv, auto_seam_rotation, apply_seam_rotation
from .ppi import ppi_residue_points_uv, ppi_atom_cloud_uv
from .viz import plot_map, PlotSpec

log = get_logger(__name__)


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


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


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="dockmap",
        description="2D surface map of docking sites + PPI overlay",
        formatter_class=_HelpFormatter,
    )

    # -----------------------
    # Inputs
    # -----------------------
    g_in = ap.add_argument_group("Inputs")
    g_in.add_argument("--protein", required=True, help="Protein coordinates PDB (ATOM/HETATM).")
    g_in.add_argument(
        "--peptides",
        required=True,
        help="Peptide poses PDB. Multiple poses separated by END / ENDMDL / MODEL blocks.",
    )
    g_in.add_argument(
        "--scores",
        required=True,
        help="Scores text file: one value per pose, in the same order as poses in --peptides.",
    )
    g_in.add_argument(
        "--ppi-file",
        required=True,
        help="Interface residue list file (one residue per line like: 'Chain  X  RES  123').",
    )
    g_in.add_argument(
        "--align-protein",
        default="none",
        choices=["none"],
        help="Protein alignment mode. Currently only 'none' (no alignment).",
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

    # -----------------------
    # Output
    # -----------------------
    g_out = ap.add_argument_group("Output")
    g_out.add_argument("--out-prefix", default="dockmap", help="Output file prefix.")
    g_out.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Figure format for the 2D map.")
    g_out.add_argument("--write-csv", action="store_true", default=True, help="Write CSV outputs (poses and PPI UV).")

    g_out.add_argument("--export-mesh", action="store_true", default=False, help="Export the QuickSurf surface mesh.")
    g_out.add_argument("--mesh-format", default="ply", choices=["obj", "ply", "stl"], help="Mesh export format.")
    g_out.add_argument("--mesh-path", default=None, help="Explicit mesh output path (optional).")
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
        default="density",
        choices=["scatter", "density", "hexbin", "trace"],
        help=(
            "How peptide poses are drawn on the 2D map. "
            "Choices: "
            "'scatter' = plot one marker per pose (best for small N or top-N subsets); "
            "'density' = smooth heatmap on a regular lon/lat grid (good default for many poses); "
            "'hexbin' = hexagonal bin counts (crisper binned view, less smoothing than density); "
            "'trace' = draw peptide backbone trace (Cα atoms + connecting line) for selected pose(s)."
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
            "How poses are weighted when aggregating into 'density' or 'hexbin' layers (ignored for scatter/trace). "
            "Choices: "
            "'none' = all poses contribute equally; "
            "'linear' = linearly rescale weights by score (emphasizes better scores); "
            "'exp' = exponential weight from Vina score (strongly emphasizes best scores)."
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
    configure_logging(args.verbose)

    log.info("dockmap pipeline start")
    log.debug("Arguments: %s", vars(args))

    # ---- Load inputs
    with Timer("Load protein atoms", log):
        protein_atoms = load_protein_atoms(args.protein)

    with Timer("Load peptide poses + scores", log):
        poses = load_poses(args.peptides, args.scores)
    if len(poses) == 0:
        raise SystemExit("No peptide poses loaded.")
    log.info("Loaded %d peptide poses", len(poses))

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
        pose_theta, pose_phi, pose_dist, pose_ids, scores = [], [], [], [], []
        for pose in poses:
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
            pose_ids.append(pose.pose_id)
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

    # ---- Pose labels selection (scatter/trace only)
    pose_labels: list[str] | None = None
    if args.pose_layer in {"scatter", "trace"} and args.pose_label != "none":
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

    if args.pose_layer == "trace":
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
        pose_layer=args.pose_layer,
        weight_mode=args.weight,
        out_format=args.format,
        background=args.background,
    )



    # Choose which scalar to send to viz for background:
    # - viz can plot normalized or raw; it will handle scaling, but it needs the raw field for raw mode.
    mesh_scalar_for_plot = mesh_scalar_raw if (args.background_colorbar_mode == "raw") else mesh_scalar



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
        with Timer("Write CSV outputs", log):
            pose_csv = out_prefix.with_name(out_prefix.name + "_poses_mapped.csv")
            with pose_csv.open("w", newline="") as f:
                wcsv = csv.writer(f)
                wcsv.writerow(["pose_id", "vina_score", "theta", "phi", "proj_distance"])
                for i in range(len(pose_theta)):
                    wcsv.writerow([pose_ids[i], scores[i], pose_theta[i], pose_phi[i], pose_dist[i]])
            log.info("Wrote CSV: %s", pose_csv)

            if ppi_contour_theta is not None and ppi_contour_phi is not None:
                ppi_csv = out_prefix.with_name(out_prefix.name + "_ppi_contour_mapped.csv")
                with ppi_csv.open("w", newline="") as f:
                    wcsv = csv.writer(f)
                    wcsv.writerow(["theta", "phi"])
                    for i in range(len(ppi_contour_theta)):
                        wcsv.writerow([ppi_contour_theta[i], ppi_contour_phi[i]])
                log.info("Wrote CSV: %s", ppi_csv)

            if ppi_points_theta is not None and ppi_points_phi is not None:
                ppi_csv2 = out_prefix.with_name(out_prefix.name + "_ppi_residue_points_mapped.csv")
                with ppi_csv2.open("w", newline="") as f:
                    wcsv = csv.writer(f)
                    wcsv.writerow(["theta", "phi", "label"])
                    for i in range(len(ppi_points_theta)):
                        lab = "" if (ppi_points_labels is None or i >= len(ppi_points_labels)) else ppi_points_labels[i]
                        wcsv.writerow([ppi_points_theta[i], ppi_points_phi[i], lab])
                log.info("Wrote CSV: %s", ppi_csv2)

    log.info("dockmap pipeline complete")
    return 0

