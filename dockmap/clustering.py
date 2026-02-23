from __future__ import annotations

import math

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


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Numerically stable regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    def _betacf(aa: float, bb: float, xx: float) -> float:
        max_iter = 200
        eps = 3.0e-14
        fpmin = 1.0e-300
        qab = aa + bb
        qap = aa + 1.0
        qam = aa - 1.0
        c = 1.0
        d = 1.0 - qab * xx / qap
        if abs(d) < fpmin:
            d = fpmin
        d = 1.0 / d
        h = d

        for m in range(1, max_iter + 1):
            m2 = 2 * m
            num = m * (bb - m) * xx
            den = (qam + m2) * (aa + m2)
            aa_term = num / den
            d = 1.0 + aa_term * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa_term / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            h *= d * c

            num = -(aa + m) * (qab + m) * xx
            den = (aa + m2) * (qap + m2)
            aa_term = num / den
            d = 1.0 + aa_term * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa_term / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps:
                break
        return h

    ln_beta = float(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    front = float(math.exp(a * math.log(x) + b * math.log(1.0 - x) - ln_beta))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def welch_ttest_two_sided_pvalue(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Two-sided Welch's t-test p-value between two score samples."""
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    n1 = int(a.size)
    n2 = int(b.size)
    if n1 == 0 or n2 == 0:
        return float("nan")

    m1 = float(np.mean(a))
    m2 = float(np.mean(b))
    v1 = float(np.var(a, ddof=1)) if n1 > 1 else 0.0
    v2 = float(np.var(b, ddof=1)) if n2 > 1 else 0.0

    se2 = (v1 / n1) + (v2 / n2)
    if se2 == 0.0:
        return 1.0 if m1 == m2 else 0.0

    t_abs = abs(m1 - m2) / float(math.sqrt(se2))

    num = se2 * se2
    den = 0.0
    if n1 > 1:
        den += (v1 * v1) / ((n1 * n1) * (n1 - 1))
    if n2 > 1:
        den += (v2 * v2) / ((n2 * n2) * (n2 - 1))
    if den == 0.0:
        return float("nan")

    dof = num / den
    if dof <= 0.0:
        return float("nan")

    x = dof / (dof + t_abs * t_abs)
    return float(_regularized_incomplete_beta(0.5 * dof, 0.5, x))


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
    scores_by_raw: dict[int, np.ndarray] = {}
    for raw in unique:
        idx = np.flatnonzero(raw_labels == raw)
        c_scores = scores[idx]
        scores_by_raw[int(raw)] = c_scores
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
    reference_info = min(infos, key=lambda d: float(d["vina_avg"]))
    reference_raw = int(reference_info["raw"])
    reference_scores = scores_by_raw[reference_raw]
    for cid, info in enumerate(infos, start=1):
        raw = int(info["raw"])
        p_value = welch_ttest_two_sided_pvalue(reference_scores, scores_by_raw[raw])
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
                "p_value": float(p_value),
                "theta_centroid": float(info["centroid_theta"]),
                "phi_centroid": float(info["centroid_phi"]),
            }
        )

    return ordered_idx, cluster_ids, summaries
