from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import numpy as np

from .util import Mesh, save_mesh_npz, load_mesh_npz, get_logger
from .io import AtomRecord

log = get_logger(__name__)


@dataclass(frozen=True)
class QuickSurfSpec:
    radius_scale: float = 1.4
    density_isovalue: float = 1.0
    grid_spacing: float = 1.0
    surface_quality: str = "max"   # max|med|low
    cache_path: Path | None = None


@dataclass(frozen=True)
class FieldInfo:
    rho: np.ndarray       # (nx,ny,nz) float32
    origin: np.ndarray    # (3,) float
    spacing: float


def _vdw_radius(element: str) -> float:
    vdw = {
        "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
        "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98
    }
    return float(vdw.get(element.upper(), 1.70))


def _interp(p0: np.ndarray, p1: np.ndarray, v0: float, v1: float, iso: float) -> np.ndarray:
    dv = (v1 - v0)
    if abs(dv) < 1e-12:
        t = 0.5
    else:
        t = (iso - v0) / dv
    return p0 + t * (p1 - p0)


def _marching_tetrahedra(volume: np.ndarray, iso: float, origin: np.ndarray, spacing: float) -> Mesh:
    nx, ny, nz = volume.shape
    verts: list[np.ndarray] = []
    faces: list[list[int]] = []

    cube_corners = np.array([
        [0,0,0], [1,0,0], [1,1,0], [0,1,0],
        [0,0,1], [1,0,1], [1,1,1], [0,1,1],
    ], dtype=int)

    tets = [
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 3, 4, 6],
        [3, 4, 6, 7],
        [1, 4, 5, 6],
        [4, 6, 7, 5],
    ]
    tet_edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

    def tet_triangles(p: np.ndarray, val: np.ndarray) -> list[list[np.ndarray]]:
        inside = val >= iso
        n_in = int(np.sum(inside))
        if n_in == 0 or n_in == 4:
            return []
        pts = []
        for a,b in tet_edges:
            ina, inb = inside[a], inside[b]
            if ina != inb:
                pts.append(_interp(p[a], p[b], float(val[a]), float(val[b]), iso))
        if len(pts) == 3:
            return [[pts[0], pts[1], pts[2]]]
        if len(pts) == 4:
            return [[pts[0], pts[1], pts[2]], [pts[0], pts[2], pts[3]]]
        return []

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                idxs = cube_corners + np.array([i, j, k], dtype=int)
                vals = volume[idxs[:,0], idxs[:,1], idxs[:,2]]
                if (vals.max() < iso) or (vals.min() > iso):
                    continue
                pos = origin + spacing * idxs.astype(float)
                for tet in tets:
                    p = pos[tet]
                    v = vals[tet]
                    tris = tet_triangles(p, v)
                    for tri in tris:
                        base = len(verts)
                        verts.extend(tri)
                        faces.append([base, base+1, base+2])

    if not verts:
        return Mesh(vertices=np.zeros((0,3), float), faces=np.zeros((0,3), int))

    vertices = np.vstack(verts).astype(float)
    faces = np.array(faces, dtype=int)
    return Mesh(vertices=vertices, faces=faces)


def build_quicksurf_mesh(
    protein_atoms: list[AtomRecord],
    spec: QuickSurfSpec,
    force_rebuild: bool = False,
    return_field: bool = False,
):
    """
    Return Mesh, and optionally FieldInfo (rho/origin/spacing) for density sampling.
    """
    if spec.cache_path and spec.cache_path.exists() and not force_rebuild and not return_field:
        mesh, meta = load_mesh_npz(spec.cache_path)
        log.debug("Loaded cached surface mesh: %s (meta=%s)", spec.cache_path, meta)
        return mesh

    coords = np.array([a.coord for a in protein_atoms], dtype=float)
    elems = [a.element for a in protein_atoms]
    radii = np.array([_vdw_radius(e) * spec.radius_scale for e in elems], dtype=float)
    spacing = float(spec.grid_spacing)

    if spec.surface_quality == "max":
        cutoff_sigma = 3.0
    elif spec.surface_quality == "med":
        cutoff_sigma = 2.5
    else:
        cutoff_sigma = 2.0

    pad = float(np.max(radii) + 3.0 * spacing)
    lo = coords.min(axis=0) - pad
    hi = coords.max(axis=0) + pad

    nxyz = np.ceil((hi - lo) / spacing).astype(int) + 1
    nx, ny, nz = map(int, nxyz)

    log.debug("Surface grid: nx=%d ny=%d nz=%d spacing=%.3f origin=%s", nx, ny, nz, spacing, lo)

    rho = np.zeros((nx, ny, nz), dtype=np.float32)
    sigmas = radii / 2.5

    xs = lo[0] + np.arange(nx) * spacing
    ys = lo[1] + np.arange(ny) * spacing
    zs = lo[2] + np.arange(nz) * spacing

    # Accumulate gaussian densities
    for (x0, y0, z0), sigma in zip(coords, sigmas):
        if sigma <= 1e-6:
            continue
        rcut = cutoff_sigma * sigma

        ix0 = int(np.floor((x0 - rcut - lo[0]) / spacing))
        ix1 = int(np.ceil((x0 + rcut - lo[0]) / spacing))
        iy0 = int(np.floor((y0 - rcut - lo[1]) / spacing))
        iy1 = int(np.ceil((y0 + rcut - lo[1]) / spacing))
        iz0 = int(np.floor((z0 - rcut - lo[2]) / spacing))
        iz1 = int(np.ceil((z0 + rcut - lo[2]) / spacing))

        ix0 = max(ix0, 0); iy0 = max(iy0, 0); iz0 = max(iz0, 0)
        ix1 = min(ix1, nx - 1); iy1 = min(iy1, ny - 1); iz1 = min(iz1, nz - 1)

        x = xs[ix0:ix1+1]
        y = ys[iy0:iy1+1]
        z = zs[iz0:iz1+1]

        gx = np.exp(-0.5 * ((x - x0) / sigma) ** 2).astype(np.float32)
        gy = np.exp(-0.5 * ((y - y0) / sigma) ** 2).astype(np.float32)
        gz = np.exp(-0.5 * ((z - z0) / sigma) ** 2).astype(np.float32)

        rho[ix0:ix1+1, iy0:iy1+1, iz0:iz1+1] += gx[:, None, None] * gy[None, :, None] * gz[None, None, :]

    mesh = _marching_tetrahedra(
        volume=rho,
        iso=float(spec.density_isovalue),
        origin=lo.astype(float),
        spacing=spacing,
    )

    log.debug("Surface mesh: vertices=%d faces=%d", mesh.vertices.shape[0], mesh.faces.shape[0])

    if spec.cache_path and not return_field:
        meta = {
            "radius_scale": spec.radius_scale,
            "density_isovalue": spec.density_isovalue,
            "grid_spacing": spec.grid_spacing,
            "surface_quality": spec.surface_quality,
            "extractor": "marching_tetrahedra",
        }
        save_mesh_npz(mesh, spec.cache_path, meta=meta)
        log.debug("Cached surface mesh -> %s", spec.cache_path)

    if return_field:
        return mesh, FieldInfo(rho=rho, origin=lo.astype(float), spacing=spacing)
    return mesh


def sample_field_trilinear(field: FieldInfo, points_xyz: np.ndarray) -> np.ndarray:
    """Trilinear sampling of rho at arbitrary XYZ points."""
    rho = field.rho
    o = field.origin
    s = float(field.spacing)

    g = (points_xyz - o[None, :]) / s
    x, y, z = g[:, 0], g[:, 1], g[:, 2]
    nx, ny, nz = rho.shape

    x = np.clip(x, 0, nx - 2)
    y = np.clip(y, 0, ny - 2)
    z = np.clip(z, 0, nz - 2)

    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); z0 = np.floor(z).astype(int)
    x1 = x0 + 1; y1 = y0 + 1; z1 = z0 + 1

    xd = x - x0; yd = y - y0; zd = z - z0

    c000 = rho[x0, y0, z0]
    c100 = rho[x1, y0, z0]
    c010 = rho[x0, y1, z0]
    c110 = rho[x1, y1, z0]
    c001 = rho[x0, y0, z1]
    c101 = rho[x1, y0, z1]
    c011 = rho[x0, y1, z1]
    c111 = rho[x1, y1, z1]

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    c = c0 * (1 - zd) + c1 * zd
    return c.astype(float)

