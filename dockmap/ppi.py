# dockmap/ppi.py
from __future__ import annotations

import numpy as np
from typing import List, Tuple
from .io import AtomRecord, ResidueID
from .util import Mesh, get_logger
from .project import project_point_to_surface_nearest, project_points_to_surface_nearest
from .mapproj import surface_point_to_spherical_uv

log = get_logger(__name__)


def _group_atoms_by_residue(atoms: list[AtomRecord]) -> dict[tuple[str, int, str], list[AtomRecord]]:
    d: dict[tuple[str, int, str], list[AtomRecord]] = {}
    for a in atoms:
        d.setdefault((a.chain, a.resseq, a.icode), []).append(a)
    return d


def _residue_point(atoms: list[AtomRecord], mode: str) -> np.ndarray:
    mode = mode.lower()
    coords = np.array([a.coord for a in atoms], float)
    if mode == "ca":
        for a in atoms:
            if a.name.upper() == "CA":
                return a.coord
        return coords.mean(axis=0)
    if mode == "res_com":
        return coords.mean(axis=0)
    if mode == "sc_com":
        sc = [a.coord for a in atoms if a.name.upper() not in {"N", "CA", "C", "O", "OXT"}]
        if sc:
            return np.array(sc, float).mean(axis=0)
        return coords.mean(axis=0)
    raise ValueError(f"Unknown residue point mode: {mode}")


def ppi_residue_points_uv(
    protein_atoms: list[AtomRecord],
    ppi: set[ResidueID],
    mesh: Mesh,
    center: np.ndarray,
    residue_point_mode: str = "sc_com",
):
    byres = _group_atoms_by_residue(protein_atoms)
    thetas, phis, labels = [], [], []

    for rid in ppi:
        atoms = byres.get((rid.chain, rid.resseq, rid.icode), [])
        if not atoms:
            continue

        rp = _residue_point(atoms, residue_point_mode)
        pr = project_point_to_surface_nearest(rp, mesh)
        th, ph = surface_point_to_spherical_uv(pr.point, center)

        thetas.append(th)
        phis.append(ph)

        # Label from interface file identity (3-letter + residue number + optional insertion code)
        # Label: residue name from protein PDB atoms + resseq (+ insertion code if any)
        resname = getattr(atoms[0], "resname", "UNK")
        icode = rid.icode or ""
        labels.append(f"{resname}{rid.resseq}{icode}")


    log.debug("PPI residue_points mapped: %d points", len(thetas))
    return np.array(thetas, float), np.array(phis, float), labels


def ppi_atom_cloud_uv(
    protein_atoms: list[AtomRecord],
    ppi: set[ResidueID],
    mesh: Mesh,
    center: np.ndarray,
    atom_filter: str = "near_surface",
    near_surface_eps: float = 1.5,
):
    """
    Fast implementation: batch-project all candidate atoms to surface once.
    """
    byres = _group_atoms_by_residue(protein_atoms)
    atom_filter = atom_filter.lower()

    # Gather candidate heavy-atom coordinates
    coords = []
    for rid in ppi:
        atoms = byres.get((rid.chain, rid.resseq, rid.icode), [])
        if not atoms:
            continue
        heavy = [a for a in atoms if a.element.upper() != "H"]
        if atom_filter in ("all_heavy", "near_surface"):
            coords.extend([a.coord for a in heavy])
        elif atom_filter == "sasa":
            raise NotImplementedError("SASA filter not implemented yet.")
        else:
            raise ValueError(f"Unknown atom_filter: {atom_filter}")

    if not coords:
        log.debug("PPI atom_cloud mapped: 0 atoms (no candidate coordinates)")
        return np.zeros((0,), float), np.zeros((0,), float)

    coords = np.asarray(coords, dtype=float)
    # Batch closest-point projection
    proj, dist, _face = project_points_to_surface_nearest(coords, mesh)

    if atom_filter == "near_surface":
        keep = dist <= float(near_surface_eps)
        proj = proj[keep]
        dist = dist[keep]

    # Convert projected points to spherical UV
    thetas = np.empty((proj.shape[0],), dtype=float)
    phis = np.empty((proj.shape[0],), dtype=float)
    for i in range(proj.shape[0]):
        thetas[i], phis[i] = surface_point_to_spherical_uv(proj[i], center)

    log.debug("PPI atom_cloud mapped: %d atoms", proj.shape[0])
    return thetas, phis

