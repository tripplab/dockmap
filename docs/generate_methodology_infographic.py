import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def add_box(ax, x, y, w, h, title, lines, facecolor, edgecolor="#2c3e50"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.02,
        y + h - 0.055,
        title,
        fontsize=12,
        fontweight="bold",
        color="#1f2d3d",
        va="top",
    )
    ax.text(
        x + 0.02,
        y + h - 0.105,
        "\n".join(f"• {line}" for line in lines),
        fontsize=10,
        color="#263238",
        va="top",
        linespacing=1.3,
    )


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="Simple,tail_width=0.7,head_width=8,head_length=10",
        color="#4a6572",
        linewidth=1.2,
    )
    ax.add_patch(arrow)


fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

fig.patch.set_facecolor("#f8fafc")
ax.set_facecolor("#f8fafc")

ax.text(
    0.5,
    0.95,
    "dockmap methodological workflow",
    ha="center",
    va="center",
    fontsize=24,
    fontweight="bold",
    color="#0f172a",
)
ax.text(
    0.5,
    0.91,
    "Input → Transformations → Output",
    ha="center",
    va="center",
    fontsize=13,
    color="#334155",
)

# Input lane
add_box(
    ax,
    0.04,
    0.17,
    0.25,
    0.65,
    "1) Input",
    [
        "Protein PDB",
        "Peptide poses PDB",
        "Docking scores",
        "PPI residue list",
    ],
    facecolor="#dbeafe",
)

# Transformation lane (split into 3 stacked stages)
add_box(
    ax,
    0.36,
    0.59,
    0.28,
    0.23,
    "2A) Surface & pose mapping",
    [
        "Build QuickSurf-like density mesh",
        "Compute peptide COM/COG per pose",
        "Project pose center to protein surface",
    ],
    facecolor="#dcfce7",
)
add_box(
    ax,
    0.36,
    0.33,
    0.28,
    0.22,
    "2B) Coordinate transforms",
    [
        "Convert mapped points to (theta, phi)",
        "Apply 2D projection (equirect / mollweide / hammer)",
        "Cluster poses by spherical distance",
    ],
    facecolor="#dcfce7",
)
add_box(
    ax,
    0.36,
    0.08,
    0.28,
    0.21,
    "2C) Overlay & weighting",
    [
        "Map PPI residues to 2D footprint",
        "Generate contour/points overlay",
        "Apply score weights + optional density/trace layer",
    ],
    facecolor="#dcfce7",
)

# Output lane
add_box(
    ax,
    0.71,
    0.17,
    0.25,
    0.65,
    "3) Output",
    [
        "2D docking map figure (PNG/PDF/SVG)",
        "Mapped pose CSV",
        "Cluster summary CSV",
        "PPI contour & residue-point CSV",
        "Optional surface mesh export (OBJ/PLY/STL)",
    ],
    facecolor="#fee2e2",
)

add_arrow(ax, (0.30, 0.495), (0.35, 0.495))
add_arrow(ax, (0.65, 0.495), (0.70, 0.495))

# Vertical flow arrows in transformation lane
add_arrow(ax, (0.50, 0.58), (0.50, 0.56))
add_arrow(ax, (0.50, 0.32), (0.50, 0.30))

ax.text(
    0.5,
    0.02,
    "dockmap: from raw docking inputs to interpretable 2D surface maps",
    ha="center",
    va="bottom",
    fontsize=10,
    color="#475569",
)

out_path = "docs/dockmap_methodology_infographic.png"
plt.tight_layout()
plt.savefig(out_path, bbox_inches="tight")
print(out_path)
