from __future__ import annotations

import numpy as np
from .util import wrap_angle, get_logger

log = get_logger(__name__)


def surface_point_to_spherical_uv(p: np.ndarray, center: np.ndarray) -> tuple[float, float]:
    v = p - center
    r = np.linalg.norm(v)
    if r < 1e-12:
        return 0.0, np.pi / 2
    theta = np.arctan2(v[1], v[0])
    phi = np.arccos(np.clip(v[2] / r, -1.0, 1.0))  # colatitude
    return float(theta), float(phi)


def apply_seam_rotation(theta: np.ndarray, rotate_rad: float) -> np.ndarray:
    return wrap_angle(theta + rotate_rad)


def auto_seam_rotation(theta: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Choose seam rotation so mean direction sits near 0 (seam away from cluster)."""
    if weights is None:
        weights = np.ones_like(theta, dtype=float)
    s = np.sum(weights * np.sin(theta))
    c = np.sum(weights * np.cos(theta))
    mean = np.arctan2(s, c)
    return float(-mean)


def uv_to_lonlat(theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = theta
    lat = (np.pi / 2.0) - phi
    return lon, lat


def project_to_2d(theta: np.ndarray, phi: np.ndarray, map_name: str) -> tuple[np.ndarray, np.ndarray]:
    map_name = map_name.lower()
    lon, lat = uv_to_lonlat(theta, phi)

    if map_name in ("equirect", "equirectangular"):
        return lon, lat

    if map_name == "hammer":
        z = np.sqrt(1 + np.cos(lat) * np.cos(lon / 2))
        x = (2 * np.sqrt(2) * np.cos(lat) * np.sin(lon / 2)) / z
        y = (np.sqrt(2) * np.sin(lat)) / z
        return x, y

    if map_name == "mollweide":
        lat_clip = np.clip(lat, -np.pi/2 + 1e-9, np.pi/2 - 1e-9)
        t = lat_clip.copy()
        rhs = np.pi * np.sin(lat_clip)
        for _ in range(10):
            f = 2*t + np.sin(2*t) - rhs
            fp = 2 + 2*np.cos(2*t)
            t = t - f / fp
        x = (2 * np.sqrt(2) / np.pi) * lon * np.cos(t)
        y = np.sqrt(2) * np.sin(t)
        return x, y

    raise ValueError(f"Unknown map projection: {map_name}")

