"""Render the three-leaflet valve from sim/src/wheatley/geometry.py for visual comparison with
McKee et al. 2026, Fig. 4b (page 7) and Fig. 5.  Top row: our model; bottom row: its mirror image
(y -> -y), i.e. the opposite handedness.  Output: sim/out/fig4b_compare.png (not versioned)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from wheatley.geometry import leaflet_grid

views = [(35, -60), (35, 30), (60, 120), (90, -90)]
fig = plt.figure(figsize=(16, 8))
for row, mirror in enumerate([False, True]):
    for col, (el, az) in enumerate(views):
        ax = fig.add_subplot(2, 4, 4 * row + col + 1, projection="3d")
        for k in range(3):
            for P, colr in zip(leaflet_grid(40, 20, k), ["gold", "olive"]):
                X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
                if mirror:
                    Y = -Y
                ax.plot_surface(X, Y, Z, color=colr, alpha=0.9, linewidth=0)
        ax.view_init(el, az)
        ax.set_box_aspect((2, 2, 1))
        ax.set_axis_off()
        ax.set_title(("mirror " if mirror else "model ") + f"el={el} az={az}", fontsize=8)
plt.tight_layout()
plt.savefig("sim/out/fig4b_compare.png", dpi=90)
