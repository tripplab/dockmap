from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

from .mapproj import project_to_2d
from .util import get_logger

log = get_logger(__name__)


BackgroundCbarLocation = Literal["right", "bottom"]
BackgroundCbarMode = Literal["norm", "raw"]


@dataclass(frozen=True)
class PlotSpec:
    map_name: str = "mollweide"
    pose_layer: str = "scatter"   # scatter|density|hexbin|trace|centroid
    weight_mode: str = "exp"      # none|exp|linear
    background: str = "none"      # none|curvature|radial
    out_format: str = "png"
    dpi: int = 300


def _convex_hull_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """2D convex hull indices (monotone chain) in CCW order."""
    pts = np.column_stack((x, y))
    n = int(pts.shape[0])
    if n <= 2:
        return np.arange(n, dtype=int)

    order = np.lexsort((pts[:, 1], pts[:, 0]))
    p = pts[order]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[int] = []
    for i in range(n):
        while len(lower) >= 2 and cross(p[lower[-2]], p[lower[-1]], p[i]) <= 0.0:
            lower.pop()
        lower.append(i)

    upper: list[int] = []
    for i in range(n - 1, -1, -1):
        while len(upper) >= 2 and cross(p[upper[-2]], p[upper[-1]], p[i]) <= 0.0:
            upper.pop()
        upper.append(i)

    hull_local = lower[:-1] + upper[:-1]
    hull_sorted = order[np.array(hull_local, dtype=int)]
    if hull_sorted.size == 0:
        return np.arange(n, dtype=int)
    return hull_sorted


def _cluster_colors(cluster_ids: np.ndarray, single_color: str | None) -> dict[int, tuple[float, float, float, float] | str]:
    unique = sorted({int(c) for c in cluster_ids.tolist()})
    if not unique:
        return {}
    if single_color:
        return {cid: single_color for cid in unique}
    cmap = cm.get_cmap("coolwarm")
    n = len(unique)
    if n == 1:
        return {unique[0]: cmap(1.0)}
    return {cid: cmap(1.0 - (i / (n - 1))) for i, cid in enumerate(unique)}


def _weights_from_scores(scores: np.ndarray | None, mode: str) -> np.ndarray | None:
    if scores is None:
        return None
    mode = mode.lower()
    if mode == "none":
        return None
    if mode == "linear":
        s = -scores
        s = s - np.min(s) + 1e-6
        return s
    if mode == "exp":
        return np.exp(-scores)
    raise ValueError(f"Unknown weight mode: {mode}")


def _wrap_lonlat(lon: np.ndarray, lat: np.ndarray, scalar: np.ndarray | None = None):
    lon2 = np.concatenate([lon, lon - 2 * np.pi, lon + 2 * np.pi])
    lat2 = np.concatenate([lat, lat, lat])
    if scalar is None:
        return lon2, lat2, None
    sc2 = np.concatenate([scalar, scalar, scalar])
    return lon2, lat2, sc2


def _gaussian_blur_fft(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """Simple gaussian blur via FFT (no SciPy)."""
    if sigma_px <= 0:
        return img
    ny, nx = img.shape
    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.fftfreq(nx) * nx
    KX, KY = np.meshgrid(kx, ky)
    H = np.exp(-0.5 * (KX**2 + KY**2) / (sigma_px**2))
    out = np.fft.ifft2(np.fft.fft2(img) * H).real
    return out


def _background_grid_from_vertices(
    lon_v: np.ndarray,
    lat_v: np.ndarray,
    sc_v: np.ndarray,
    lon_bins: int = 360,
    lat_bins: int = 180,
):
    """
    Bin vertices into lon/lat grid, compute mean scalar per bin.
    Return grid and mask for populated bins.
    """
    lon_edges = np.linspace(-np.pi, np.pi, lon_bins + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, lat_bins + 1)

    ix = np.searchsorted(lon_edges, lon_v, side="right") - 1
    iy = np.searchsorted(lat_edges, lat_v, side="right") - 1
    ix = np.clip(ix, 0, lon_bins - 1)
    iy = np.clip(iy, 0, lat_bins - 1)

    sum_grid = np.zeros((lat_bins, lon_bins), float)
    cnt_grid = np.zeros((lat_bins, lon_bins), float)
    np.add.at(sum_grid, (iy, ix), sc_v)
    np.add.at(cnt_grid, (iy, ix), 1.0)

    Z = np.zeros_like(sum_grid)
    mask = cnt_grid > 0
    Z[mask] = sum_grid[mask] / cnt_grid[mask]

    lon_cent = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_cent = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    LON, LAT = np.meshgrid(lon_cent, lat_cent)
    return LON, LAT, Z, mask


def _default_background_label(background: str, mode: BackgroundCbarMode) -> str:
    b = background.lower()
    if b == "curvature":
        base = "Curvature (proxy)"
    elif b == "radial":
        base = "Radial distance"
    else:
        base = "Background scalar"
    if mode == "norm":
        return f"{base} (normalized)"
    return base


def _draw_background(
    ax,
    map_name: str,
    mesh_theta: np.ndarray,
    mesh_phi: np.ndarray,
    mesh_scalar: np.ndarray,
    *,
    mode: BackgroundCbarMode = "norm",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Draw background shading from mesh vertices and return a matplotlib "mappable"
    suitable for fig.colorbar(...). Colorbar scaling must match what is plotted.

    mode:
      - "norm": normalize populated bins to [0,1] before plotting
      - "raw": plot raw scalar values with Normalize(vmin, vmax) and levels spanning [vmin, vmax]
    """
    lon = mesh_theta
    lat = (np.pi / 2) - mesh_phi
    lon_w, lat_w, sc_w = _wrap_lonlat(lon, lat, mesh_scalar)

    LON, LAT, Z, mask = _background_grid_from_vertices(lon_w, lat_w, sc_w)
    if not np.any(mask):
        return None

    Zm = np.ma.masked_where(~mask, Z)

    # Decide plot range + levels so contourf + colorbar obey vmin/vmax
    if mode == "norm":
        zmin = float(np.nanmin(Zm))
        zmax = float(np.nanmax(Zm))
        if zmax > zmin:
            Zplot = (Zm - zmin) / (zmax - zmin)
        else:
            Zplot = np.zeros_like(Zm)

        vmin_plot, vmax_plot = 0.0, 1.0
        norm = Normalize(vmin=vmin_plot, vmax=vmax_plot)
        levels_filled = np.linspace(vmin_plot, vmax_plot, 41)   # 40 bands
        levels_lines = np.linspace(vmin_plot, vmax_plot, 13)    # 12 lines

    else:
        # raw mode: use user-provided vmin/vmax if present, else infer from populated bins
        if vmin is None:
            vmin = float(np.nanmin(Zm))
        if vmax is None:
            vmax = float(np.nanmax(Zm))
        if vmax <= vmin:
            vmax = vmin + 1e-12

        vmin_plot, vmax_plot = float(vmin), float(vmax)
        norm = Normalize(vmin=vmin_plot, vmax=vmax_plot)
        Zplot = Zm

        # IMPORTANT: levels must span [vmin, vmax] so the colorbar matches the user range
        levels_filled = np.linspace(vmin_plot, vmax_plot, 41)  # 40 bands
        levels_lines = np.linspace(vmin_plot, vmax_plot, 13)   # 12 lines

    TH = LON
    PH = (np.pi / 2) - LAT
    X, Y = project_to_2d(TH, PH, map_name)

    mappable = ax.contourf(
        X,
        Y,
        Zplot,
        levels=levels_filled,
        cmap=cmap,
        norm=norm,
        alpha=0.70,
        zorder=0,
        extend="both" if mode == "raw" else "neither",
    )
    ax.contour(
        X,
        Y,
        Zplot,
        levels=levels_lines,
        linewidths=0.6,
        alpha=0.7,
        zorder=0,
    )
    return mappable


def _draw_pose_labels(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str] | None,
    *,
    dy: float = 0.015,
    fontsize: int = 7,
    zorder: int = 6,
):
    """
    Draw labels above pose markers (scatter/trace only).
    labels must align with x/y (empty string => skip).
    """
    if labels is None:
        return
    n = min(len(labels), len(x))
    for i in range(n):
        lab = labels[i]
        if not lab:
            continue
        ax.text(
            float(x[i]),
            float(y[i] + dy),
            lab,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            zorder=zorder,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.70),
        )


def plot_map(
    pose_theta: np.ndarray,
    pose_phi: np.ndarray,
    pose_scores: np.ndarray | None,
    out_path: str | Path,
    plot_spec: PlotSpec,
    ppi_contour_theta: np.ndarray | None = None,
    ppi_contour_phi: np.ndarray | None = None,
    ppi_points_theta: np.ndarray | None = None,
    ppi_points_phi: np.ndarray | None = None,
    ppi_points_labels: list[str] | None = None,
    mesh_theta: np.ndarray | None = None,
    mesh_phi: np.ndarray | None = None,
    mesh_scalar: np.ndarray | None = None,
    pose_labels: list[str] | None = None,
    trace_lines: list[tuple[np.ndarray, np.ndarray]] | None = None,   # [(theta_i, phi_i), ...]
    trace_labels: list[str] | None = None,                            # optional label per trace (same order)
    cluster_ids: np.ndarray | None = None,
    cluster_theta: np.ndarray | None = None,
    cluster_phi: np.ndarray | None = None,
    cluster_contour: str = "none",
    cluster_contour_color: str | None = None,
    background_colorbar: bool = False,
    background_colorbar_location: BackgroundCbarLocation = "right",
    background_colorbar_mode: BackgroundCbarMode = "norm",
    background_colorbar_label: str | None = None,
    background_colorbar_vmin: float | None = None,
    background_colorbar_vmax: float | None = None,
):
    map_name = plot_spec.map_name.lower()
    lon = pose_theta
    lat = (np.pi / 2) - pose_phi
    w = _weights_from_scores(pose_scores, plot_spec.weight_mode)

    fig = plt.figure(figsize=(10, 5.2))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Docking site map ({map_name})")

    # Background
    bg_mappable = None
    if plot_spec.background != "none":
        if mesh_theta is None or mesh_phi is None or mesh_scalar is None:
            raise ValueError("Background requested but mesh_theta/mesh_phi/mesh_scalar not provided.")
        bg_mappable = _draw_background(
            ax,
            map_name,
            mesh_theta,
            mesh_phi,
            mesh_scalar,
            mode=background_colorbar_mode,
            cmap="viridis",
            vmin=background_colorbar_vmin,
            vmax=background_colorbar_vmax,
        )

        if background_colorbar and bg_mappable is not None:
            label = background_colorbar_label or _default_background_label(plot_spec.background, background_colorbar_mode)

            if background_colorbar_location == "bottom":
                cbar = fig.colorbar(
                    bg_mappable,
                    ax=ax,
                    orientation="horizontal",
                    fraction=0.06,
                    pad=0.08,
                )
            else:
                cbar = fig.colorbar(
                    bg_mappable,
                    ax=ax,
                    orientation="vertical",
                    fraction=0.046,
                    pad=0.04,
                )

            cbar.set_label(label)

            # For normalized mode, use simple ticks for readability
            if background_colorbar_mode == "norm":
                cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    # Wrap to avoid seam artifacts (for density)
    lon_w = np.concatenate([lon, lon - 2 * np.pi, lon + 2 * np.pi])
    lat_w = np.concatenate([lat, lat, lat])
    w_w = None if w is None else np.concatenate([w, w, w])

    # ---- Cluster contour overlay (convex hull in projected map space)
    if cluster_contour != "none":
        if cluster_ids is None:
            raise ValueError("cluster_contour requested but cluster_ids not provided.")
        ccolors = _cluster_colors(cluster_ids, cluster_contour_color)
        x_pose, y_pose = project_to_2d(lon, pose_phi, map_name)
        for cid in sorted({int(c) for c in cluster_ids.tolist()}):
            idx = np.flatnonzero(cluster_ids == cid)
            if idx.size == 0:
                continue
            col = ccolors[cid]
            if idx.size == 1:
                ax.scatter([x_pose[idx[0]]], [y_pose[idx[0]]], s=90, facecolors="none", edgecolors=[col], linewidths=1.5, zorder=3)
                continue
            hidx = _convex_hull_xy(x_pose[idx], y_pose[idx])
            hx = x_pose[idx][hidx]
            hy = y_pose[idx][hidx]
            hx = np.append(hx, hx[0])
            hy = np.append(hy, hy[0])
            if cluster_contour == "filled":
                ax.fill(hx, hy, color=col, alpha=0.12, zorder=2.2)
            ax.plot(hx, hy, color=col, linewidth=1.5, zorder=3.1)

    # ---- Pose layers
    if plot_spec.pose_layer == "scatter":
        x, y = project_to_2d(lon, pose_phi, map_name)
        ax.scatter(x, y, s=10, alpha=0.75, zorder=2)

        # Halo ring
        ax.scatter(
            x, y,
            s=180, marker="o",
            facecolors="none",
            edgecolors="white",
            linewidths=2.0,
            zorder=6,
        )
        # Inner dot
        ax.scatter(
            x, y,
            s=40, marker="o",
            edgecolors="black",
            linewidths=0.8,
            zorder=7,
        )

        _draw_pose_labels(ax, x, y, pose_labels, dy=0.03, fontsize=7, zorder=8)

    elif plot_spec.pose_layer == "trace":
        if trace_lines is None or len(trace_lines) == 0:
            raise ValueError("pose_layer=trace but trace_lines not provided.")

        # If trace_labels not provided, just don't label traces
        if trace_labels is None:
            trace_labels = [""] * len(trace_lines)

        for i, (tth, tph) in enumerate(trace_lines):
            if tth is None or tph is None or len(tth) == 0:
                continue

            tx, ty = project_to_2d(tth, tph, map_name)

            # Polyline (N->C)
            ax.plot(tx, ty, linewidth=1.6, alpha=0.95, zorder=3)

            # CA markers: halo + inner dot
            ax.scatter(
                tx, ty,
                s=140, marker="o",
                facecolors="none",
                edgecolors="white",
                linewidths=2.0,
                zorder=4,
            )
            ax.scatter(
                tx, ty,
                s=34, marker="o",
                edgecolors="black",
                linewidths=0.9,
                zorder=5,
            )

            # Label each trace (above first CA)
            lab = trace_labels[i] if i < len(trace_labels) else ""
            if lab:
                _draw_pose_labels(
                    ax,
                    np.array([tx[0]]),
                    np.array([ty[0]]),
                    [lab],
                    dy=0.03,
                    fontsize=8,
                    zorder=8,
                )

    elif plot_spec.pose_layer == "centroid":
        if cluster_theta is None or cluster_phi is None:
            raise ValueError("pose_layer=centroid but cluster_theta/cluster_phi not provided.")
        x, y = project_to_2d(cluster_theta, cluster_phi, map_name)

        ax.scatter(
            x, y,
            s=180, marker="o",
            facecolors="none",
            edgecolors="white",
            linewidths=2.0,
            zorder=6,
        )
        ax.scatter(
            x, y,
            s=40, marker="o",
            edgecolors="black",
            linewidths=0.8,
            zorder=7,
        )
        centroid_labels = [str(i + 1) for i in range(len(x))]
        _draw_pose_labels(ax, x, y, centroid_labels, dy=0.03, fontsize=8, zorder=8)

    elif plot_spec.pose_layer == "hexbin":
        x, y = project_to_2d(lon, pose_phi, map_name)
        ax.hexbin(x, y, gridsize=70, mincnt=1, zorder=2)

    elif plot_spec.pose_layer == "density":
        lon_edges = np.linspace(-np.pi, np.pi, 360 + 1)
        lat_edges = np.linspace(-np.pi / 2, np.pi / 2, 180 + 1)
        H, _, _ = np.histogram2d(lat_w, lon_w, bins=[lat_edges, lon_edges], weights=w_w)
        Hs = _gaussian_blur_fft(H, sigma_px=2.0)

        lon_cent = 0.5 * (lon_edges[:-1] + lon_edges[1:])
        lat_cent = 0.5 * (lat_edges[:-1] + lat_edges[1:])
        LON, LAT = np.meshgrid(lon_cent, lat_cent)
        TH = LON
        PH = (np.pi / 2) - LAT
        X, Y = project_to_2d(TH, PH, map_name)

        ax.contourf(X, Y, Hs, levels=20, alpha=0.90, zorder=1)

    else:
        raise ValueError(f"Unknown pose_layer: {plot_spec.pose_layer}")

    # ---- PPI overlays
    # 1) Atom-cloud contour footprint
    if ppi_contour_theta is not None and ppi_contour_phi is not None and len(ppi_contour_theta) > 0:
        p_lon = ppi_contour_theta
        p_lat = (np.pi / 2) - ppi_contour_phi

        if len(p_lon) >= 20:
            lon_edges = np.linspace(-np.pi, np.pi, 360 + 1)
            lat_edges = np.linspace(-np.pi / 2, np.pi / 2, 180 + 1)
            P, _, _ = np.histogram2d(p_lat, p_lon, bins=[lat_edges, lon_edges])
            Ps = _gaussian_blur_fft(P, sigma_px=2.0)

            lon_cent = 0.5 * (lon_edges[:-1] + lon_edges[1:])
            lat_cent = 0.5 * (lat_edges[:-1] + lat_edges[1:])
            LON, LAT = np.meshgrid(lon_cent, lat_cent)
            X, Y = project_to_2d(LON, (np.pi / 2 - LAT), map_name)
            ax.contour(X, Y, Ps, levels=5, linewidths=1.2, zorder=3)
        else:
            px, py = project_to_2d(p_lon, ppi_contour_phi, map_name)
            ax.scatter(px, py, s=18, alpha=0.9, zorder=3)

    # 2) Residue-points footprint (squares + labels)
    rx = ry = None
    if ppi_points_theta is not None and ppi_points_phi is not None and len(ppi_points_theta) > 0:
        rx, ry = project_to_2d(ppi_points_theta, ppi_points_phi, map_name)

        ax.scatter(
            rx,
            ry,
            s=160,
            marker="s",
            facecolors="none",
            edgecolors="white",
            linewidths=2.0,
            zorder=3,   # keep behind pose halos (poses use zorder 6/7)
        )
        ax.scatter(
            rx,
            ry,
            s=36,
            marker="s",
            edgecolors="black",
            linewidths=1.0,
            zorder=4,
        )

    if ppi_points_labels is not None and rx is not None and ry is not None:
        dy = 0.03
        n = min(len(ppi_points_labels), len(rx))
        for i in range(n):
            ax.text(
                float(rx[i]),
                float(ry[i] + dy),
                ppi_points_labels[i],
                ha="center",
                va="bottom",
                fontsize=7,
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )

    ax.axis("off")
    out_path = Path(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=plot_spec.dpi)
    plt.close(fig)
