from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging
import time
import numpy as np


# -----------------------------
# Logging
# -----------------------------

def configure_logging(verbose: int = 0) -> None:
    """
    verbose=0: WARNING+
    verbose=1: INFO+
    verbose>=2: DEBUG+
    """
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    # quiet noisy third-party loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("trimesh").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class Timer:
    """Simple context manager timer for verbose logs."""
    def __init__(self, label: str, logger: logging.Logger, level: int = logging.INFO):
        self.label = label
        self.logger = logger
        self.level = level
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.time()
        self.logger.log(self.level, f"{self.label} ...")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        if exc is None:
            self.logger.log(self.level, f"{self.label} done ({dt:.2f}s)")
        else:
            self.logger.log(logging.ERROR, f"{self.label} failed ({dt:.2f}s): {exc}")


# -----------------------------
# General helpers
# -----------------------------

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def wrap_angle(theta: np.ndarray) -> np.ndarray:
    """Wrap radians to [-pi, pi)."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


# -----------------------------
# Mesh container + cache
# -----------------------------

@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray  # (N,3)
    faces: np.ndarray     # (M,3) int


def save_mesh_npz(mesh: Mesh, path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    meta_json = json.dumps(meta or {})
    np.savez_compressed(path, vertices=mesh.vertices, faces=mesh.faces, meta=meta_json)


def load_mesh_npz(path: str | Path) -> tuple[Mesh, dict]:
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    vertices = z["vertices"]
    faces = z["faces"]
    meta_json = z["meta"].item() if "meta" in z else "{}"
    meta = json.loads(meta_json)
    return Mesh(vertices=vertices, faces=faces), meta


# -----------------------------
# Mesh export (dependency-free)
# -----------------------------

def write_mesh(
    mesh: Mesh,
    out_path: str | Path,
    fmt: str,
    vertex_scalar: np.ndarray | None = None,
    scalar_name: str = "scalar",
) -> Path:
    """
    Writes triangle mesh as ASCII OBJ, PLY, or STL.

    If vertex_scalar is provided (len==n_vertices), it is exported ONLY for PLY
    as an extra property with name scalar_name.
    """
    out_path = Path(out_path)
    fmt = fmt.lower().strip()
    if fmt not in {"obj", "ply", "stl"}:
        raise ValueError(f"Unsupported mesh format: {fmt}")

    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError("Mesh is empty; cannot export.")

    if vertex_scalar is not None:
        vertex_scalar = np.asarray(vertex_scalar, dtype=float)
        if vertex_scalar.shape[0] != mesh.vertices.shape[0]:
            raise ValueError("vertex_scalar must have length equal to number of vertices")
        if fmt != "ply":
            raise ValueError("Per-vertex scalar export is only supported for PLY.")

    if fmt == "obj":
        _write_obj(mesh, out_path)
    elif fmt == "ply":
        _write_ply_ascii(mesh, out_path, vertex_scalar=vertex_scalar, scalar_name=scalar_name)
    else:
        _write_stl_ascii(mesh, out_path)

    return out_path


def _write_obj(mesh: Mesh, out_path: Path) -> None:
    v = mesh.vertices
    f = mesh.faces
    lines = []
    lines.append("# dockmap quicksurf mesh\n")
    for i in range(v.shape[0]):
        lines.append(f"v {v[i,0]:.6f} {v[i,1]:.6f} {v[i,2]:.6f}\n")
    for i in range(f.shape[0]):
        a, b, c = f[i] + 1  # OBJ 1-based
        lines.append(f"f {a} {b} {c}\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def _write_ply_ascii(
    mesh: Mesh,
    out_path: Path,
    vertex_scalar: np.ndarray | None = None,
    scalar_name: str = "scalar",
) -> None:
    v = mesh.vertices
    f = mesh.faces
    header = [
        "ply\n",
        "format ascii 1.0\n",
        "comment dockmap quicksurf mesh\n",
        f"element vertex {v.shape[0]}\n",
        "property float x\n",
        "property float y\n",
        "property float z\n",
    ]
    if vertex_scalar is not None:
        header.append(f"property float {scalar_name}\n")
    header += [
        f"element face {f.shape[0]}\n",
        "property list uchar int vertex_indices\n",
        "end_header\n",
    ]

    lines = header
    if vertex_scalar is None:
        for i in range(v.shape[0]):
            lines.append(f"{v[i,0]:.6f} {v[i,1]:.6f} {v[i,2]:.6f}\n")
    else:
        for i in range(v.shape[0]):
            lines.append(f"{v[i,0]:.6f} {v[i,1]:.6f} {v[i,2]:.6f} {vertex_scalar[i]:.6f}\n")

    for i in range(f.shape[0]):
        a, b, c = f[i]
        lines.append(f"3 {a} {b} {c}\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def _write_stl_ascii(mesh: Mesh, out_path: Path) -> None:
    v = mesh.vertices
    f = mesh.faces

    def normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            return np.array([0.0, 0.0, 0.0], float)
        return n / nn

    lines = []
    lines.append("solid dockmap_quicksurf\n")
    for i in range(f.shape[0]):
        ia, ib, ic = f[i]
        a, b, c = v[ia], v[ib], v[ic]
        n = normal(a, b, c)
        lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
        lines.append("    outer loop\n")
        lines.append(f"      vertex {a[0]:.6e} {a[1]:.6e} {a[2]:.6e}\n")
        lines.append(f"      vertex {b[0]:.6e} {b[1]:.6e} {b[2]:.6e}\n")
        lines.append(f"      vertex {c[0]:.6e} {c[1]:.6e} {c[2]:.6e}\n")
        lines.append("    endloop\n")
        lines.append("  endfacet\n")
    lines.append("endsolid dockmap_quicksurf\n")
    out_path.write_text("".join(lines), encoding="utf-8")


# -----------------------------
# Background scalars + smoothing
# -----------------------------

def vertex_laplacian_magnitude(mesh: Mesh) -> np.ndarray:
    """Curvature proxy: ||Δv|| using uniform graph Laplacian."""
    v = mesh.vertices
    f = mesh.faces
    n = v.shape[0]
    nbrs = [set() for _ in range(n)]
    for a, b, c in f:
        nbrs[a].update([b, c])
        nbrs[b].update([a, c])
        nbrs[c].update([a, b])

    lap = np.zeros_like(v, dtype=float)
    for i in range(n):
        ni = list(nbrs[i])
        if not ni:
            continue
        lap[i] = v[ni].mean(axis=0) - v[i]
    return np.linalg.norm(lap, axis=1)


def radial_distance(mesh: Mesh, center: np.ndarray) -> np.ndarray:
    return np.linalg.norm(mesh.vertices - center[None, :], axis=1)


def normalize_01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if hi - lo < eps:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def smooth_scalar_neighbor_average(mesh: Mesh, scalar: np.ndarray, n_iter: int = 1) -> np.ndarray:
    """Smooth per-vertex scalar by neighbor-averaging for n_iter iterations."""
    scalar = np.asarray(scalar, dtype=float).copy()
    if n_iter <= 0:
        return scalar

    f = mesh.faces
    n = mesh.vertices.shape[0]
    nbrs = [set() for _ in range(n)]
    for a, b, c in f:
        nbrs[a].update([b, c])
        nbrs[b].update([a, c])
        nbrs[c].update([a, b])

    for _ in range(n_iter):
        new = scalar.copy()
        for i in range(n):
            ni = nbrs[i]
            if not ni:
                continue
            new[i] = float(np.mean([scalar[j] for j in ni]))
        scalar = new
    return scalar

