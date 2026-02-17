# dockmap/project.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from .util import Mesh, get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ProjectionResult:
    point: np.ndarray
    distance: float
    face_index: int | None


def _try_trimesh():
    try:
        import trimesh  # type: ignore
        return trimesh
    except Exception:
        return None


def _nearest_vertex_fallback_batch(points: np.ndarray, mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SciPy-free fallback: nearest vertex for each point.
    Returns (proj_points, distances, face_ids=-1)
    """
    points = np.asarray(points, dtype=float)
    v = mesh.vertices
    proj = np.empty((points.shape[0], 3), dtype=float)
    dist = np.empty((points.shape[0],), dtype=float)
    face = -np.ones((points.shape[0],), dtype=int)

    # For modest N (PPI atoms ~100-1000), looping points is OK.
    # Each iteration does a vectorized distance-to-vertices pass.
    for i, q in enumerate(points):
        dv = v - q[None, :]
        d2 = np.einsum("ij,ij->i", dv, dv)
        j = int(np.argmin(d2))
        proj[i] = v[j]
        dist[i] = float(np.sqrt(d2[j]))
    return proj, dist, face


def project_points_to_surface_nearest(points: np.ndarray, mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch nearest-point projection to a triangle mesh.

    Preferred: trimesh.proximity.closest_point(mesh, points) (fast when batched).
    Fallback: nearest mesh vertex (SciPy-free, robust).

    Returns:
      proj_points: (N,3)
      distances:   (N,)
      face_ids:    (N,)  (=-1 if unavailable)
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be shaped (N,3)")

    trimesh = _try_trimesh()
    if trimesh is not None:
        try:
            tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
            proj, dist, face_id = trimesh.proximity.closest_point(tm, points)
            proj = np.asarray(proj, dtype=float)
            dist = np.asarray(dist, dtype=float)
            if face_id is None:
                face = -np.ones((points.shape[0],), dtype=int)
            else:
                face = np.asarray(face_id, dtype=int)
            return proj, dist, face
        except Exception as e:
            log.debug("trimesh closest_point batch unavailable (%s); falling back to nearest-vertex.", e)

    return _nearest_vertex_fallback_batch(points, mesh)


def project_point_to_surface_nearest(q: np.ndarray, mesh: Mesh) -> ProjectionResult:
    """
    Single-point wrapper around the batch projector.
    """
    proj, dist, face = project_points_to_surface_nearest(np.asarray([q], dtype=float), mesh)
    fid = int(face[0]) if int(face[0]) >= 0 else None
    return ProjectionResult(point=proj[0], distance=float(dist[0]), face_index=fid)


def project_point_to_surface_raycast(q: np.ndarray, center: np.ndarray, mesh: Mesh) -> ProjectionResult:
    """Raycast from center towards q. Requires trimesh; falls back to nearest."""
    trimesh = _try_trimesh()
    if trimesh is None:
        log.debug("Raycast requested but trimesh unavailable; using nearest projection.")
        return project_point_to_surface_nearest(q, mesh)

    try:
        tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
        direction = q - center
        n = np.linalg.norm(direction)
        if n < 1e-12:
            return project_point_to_surface_nearest(q, mesh)
        direction = direction / n

        origins = np.array([center], float)
        directions = np.array([direction], float)
        locs, _idx_ray, idx_tri = tm.ray.intersects_location(
            ray_origins=origins, ray_directions=directions, multiple_hits=False
        )
        if len(locs) == 0:
            return project_point_to_surface_nearest(q, mesh)

        p = np.asarray(locs[0], float)
        return ProjectionResult(
            point=p,
            distance=float(np.linalg.norm(q - p)),
            face_index=int(idx_tri[0]) if len(idx_tri) else None,
        )
    except Exception as e:
        log.debug("Raycast failed (%s); falling back to nearest.", e)
        return project_point_to_surface_nearest(q, mesh)

