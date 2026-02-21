from __future__ import annotations

import numpy as np


def spherical_to_unit(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Convert spherical angles (radians) to unit vectors, phi=colatitude."""
    st = np.sin(phi)
    x = st * np.cos(theta)
    y = st * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))


def angular_distance_matrix(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Pairwise great-circle angular distance matrix in radians."""
    u = spherical_to_unit(theta, phi)
    dots = np.clip(u @ u.T, -1.0, 1.0)
    return np.arccos(dots)


def cluster_connected_components(theta: np.ndarray, phi: np.ndarray, threshold_rad: float) -> np.ndarray:
    """
    Cluster by connectivity under angular threshold:
    points i and j share an edge if angular distance <= threshold_rad.
    Returns 0-based component ids in input order.
    """
    n = int(theta.shape[0])
    if n == 0:
        return np.empty((0,), dtype=int)

    dist = angular_distance_matrix(theta, phi)
    adj = dist <= float(threshold_rad)
    np.fill_diagonal(adj, True)

    labels = np.full((n,), -1, dtype=int)
    comp = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        stack = [i]
        labels[i] = comp
        while stack:
            cur = stack.pop()
            neigh = np.flatnonzero(adj[cur] & (labels == -1))
            if neigh.size:
                labels[neigh] = comp
                stack.extend(neigh.tolist())
        comp += 1
    return labels


def _cluster_centroid_angles(theta: np.ndarray, phi: np.ndarray) -> tuple[float, float]:
    """Spherical centroid from mean unit vector (returns theta, phi in radians)."""
    v = spherical_to_unit(theta, phi)
    m = v.mean(axis=0)
    r = float(np.linalg.norm(m))
    if r == 0.0:
        return float(theta[0]), float(phi[0])
    m = m / r
    th = float(np.arctan2(m[1], m[0]))
    ph = float(np.arccos(np.clip(m[2], -1.0, 1.0)))
    return th, ph


def reorder_clusters_and_poses(
    theta: np.ndarray,
    phi: np.ndarray,
    scores: np.ndarray,
    pose_ids: list[str],
    raw_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | str]]]:
    """
    Reassign cluster IDs by size desc, then best score asc, then original raw label.
    Return:
      - ordered_idx: indices for pose rows sorted by cluster_id then score asc then pose_id
      - cluster_ids: 1-based reassigned cluster id per input pose
      - summaries: per-cluster dicts for *_clusters.csv
    """
    unique = np.unique(raw_labels)
    infos: list[dict[str, float | int | str]] = []
    for raw in unique:
        idx = np.flatnonzero(raw_labels == raw)
        c_scores = scores[idx]
        best_local = int(np.argmin(c_scores))
        best_idx = int(idx[best_local])
        c_theta = theta[idx]
        c_phi = phi[idx]
        cth, cph = _cluster_centroid_angles(c_theta, c_phi)
        infos.append(
            {
                "raw": int(raw),
                "size": int(idx.size),
                "best_score": float(c_scores[best_local]),
                "best_pose_id": str(pose_ids[best_idx]),
                "vina_min": float(np.min(c_scores)),
                "vina_max": float(np.max(c_scores)),
                "vina_avg": float(np.mean(c_scores)),
                "vina_stddev": float(np.std(c_scores)),
                "centroid_theta": float(cth),
                "centroid_phi": float(cph),
            }
        )

    infos.sort(key=lambda d: (-int(d["size"]), float(d["best_score"]), int(d["raw"])))
    remap = {int(info["raw"]): i + 1 for i, info in enumerate(infos)}

    cluster_ids = np.array([remap[int(r)] for r in raw_labels], dtype=int)
    ordered_idx = np.array(
        sorted(range(len(pose_ids)), key=lambda i: (cluster_ids[i], float(scores[i]), pose_ids[i])), dtype=int
    )

    summaries: list[dict[str, float | int | str]] = []
    for cid, info in enumerate(infos, start=1):
        summaries.append(
            {
                "cluster_id": cid,
                "n_poses": int(info["size"]),
                "best_pose_id": str(info["best_pose_id"]),
                "best_vina_score": float(info["best_score"]),
                "vina_score_min": float(info["vina_min"]),
                "vina_score_max": float(info["vina_max"]),
                "vina_score_avg": float(info["vina_avg"]),
                "vina_score_stddev": float(info["vina_stddev"]),
                "theta_centroid": float(info["centroid_theta"]),
                "phi_centroid": float(info["centroid_phi"]),
            }
        )

    return ordered_idx, cluster_ids, summaries
