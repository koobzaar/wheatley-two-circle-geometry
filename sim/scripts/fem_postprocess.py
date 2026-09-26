"""A5 post-processing: geometric orifice area (GOA) and displacement summary.

GOA = area of the ring disk (radius R) not covered by the top-view projection of the three deformed
leaflets (leaflet 0 from the FEM, leaflets 1, 2 by rotation: the load and the clamping are symmetric
and the leaflets do not touch while opening).  The unloaded valve must give GOA ~ 0 (C-016).
Run: uv run --project sim python sim/scripts/fem_postprocess.py crease fillet
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wheatley import geometry as g  # noqa: E402


def junction_stress(r, band=1.5, zlo=0.2, zhi=0.8):
    """von Mises (cell values, reference centroids) within `band` mm of the junction M(z), for
    z/zmax in [zlo, zhi] (away from the clamped base and from the free edge)."""
    R, H = float(r["R"]), float(r["H"])
    zz = np.linspace(0, 1, 2001)
    Mj = g.junction_point(zz) * np.array([R, R, R * H])
    from scipy.spatial import cKDTree
    d, _ = cKDTree(Mj).query(r["xq"])
    zrel = r["xq"][:, 2] / (R * H)
    sel = (d < band) & (zrel > zlo) & (zrel < zhi)
    v = r["vm"][sel]
    far = r["vm"][(d > 3 * band) & (zrel > zlo) & (zrel < zhi)]
    return v.max(), np.percentile(v, 95), np.median(v), np.median(far), sel.sum()

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "fem")
FIG = os.path.join(os.path.dirname(__file__), "..", "..", "notes", "technote", "figures")


def covered_mask(tri_xy, R, n=900):
    xs = np.linspace(-R, R, n)
    h = xs[1] - xs[0]
    XX, YY = np.meshgrid(xs, xs)
    cov = np.zeros_like(XX, dtype=bool)
    for a, b, c in tri_xy:
        lo = np.floor((np.minimum(np.minimum(a, b), c) + R) / h).astype(int)
        hi = np.ceil((np.maximum(np.maximum(a, b), c) + R) / h).astype(int)
        i0, i1 = max(lo[1], 0), min(hi[1] + 1, n)
        j0, j1 = max(lo[0], 0), min(hi[0] + 1, n)
        if i0 >= i1 or j0 >= j1:
            continue
        px, py = XX[i0:i1, j0:j1], YY[i0:i1, j0:j1]
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-14:
            continue
        l1 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / d
        l2 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / d
        inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (1 - l1 - l2 >= -1e-9)
        cov[i0:i1, j0:j1] |= inside
    disk = XX**2 + YY**2 <= R**2
    return XX, YY, cov, disk, h


def goa(Xdef, tris, R, N=3, ghost=None):
    """Open (uncovered) area of the ring disk in top view.  ghost = (gX, gT): the corner removed at the base of post P
    (make_leaflet_meshes.py), kept undeformed; it is next to the clamped post and base, so its motion is neglected."""
    tri_all = []
    for k in range(N):
        a = 2 * np.pi * k / N
        Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        P = Xdef[:, :2] @ Rm.T
        tri_all.append(P[tris])
        if ghost is not None and len(ghost[1]):
            tri_all.append((ghost[0][:, :2] @ Rm.T)[ghost[1]])
    XX, YY, cov, disk, h = covered_mask(np.vstack(tri_all), R)
    open_ = disk & ~cov
    return open_.sum() * h * h, (XX, YY, open_, disk, tri_all)


if __name__ == "__main__":
    names = sys.argv[1:] or ["crease", "fillet"]
    fig, axs = plt.subplots(1, len(names) + 1, figsize=(3.2 * (len(names) + 1), 3.2))
    for j, name in enumerate(names):
        r = np.load(os.path.join(OUT, f"result_{name}.npz"))
        X, tris, R = r["X"], r["tris"], float(r["R"])
        gh = (r["ghostX"], r["ghostT"]) if "ghostX" in r.files else None
        if j == 0:
            A0, (XX, YY, op, disk, tri0) = goa(X, tris, R, ghost=gh)
            print(f"unloaded: GOA = {A0 / 100:.4f} cm^2")
            ax = axs[0]
            ax.imshow(np.where(disk, np.where(op, 1.0, 0.25), np.nan), extent=[-R, R, -R, R],
                      origin="lower", cmap="Blues", vmin=0, vmax=1)
            ax.set_title(f"unloaded, GOA {A0 / 100:.3f} cm²", fontsize=9)
            ax.set_axis_off()
        for lev, U in zip(r["levels"], r["U"]):
            A, _ = goa(X + U, tris, R, ghost=gh)
            mag = np.linalg.norm(U, axis=1)
            print(f"{name}: p = {lev:7.2f} mmHg  GOA = {A / 100:.4f} cm^2  max|u| = {mag.max():.3f} mm  "
                  f"u_z in [{U[:, 2].min():.3f}, {U[:, 2].max():.3f}] mm")
        A, (XX, YY, op, disk, _) = goa(X + r["U"][-1], tris, R, ghost=gh)
        ax = axs[j + 1]
        ax.imshow(np.where(disk, np.where(op, 1.0, 0.25), np.nan), extent=[-R, R, -R, R],
                  origin="lower", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{name}, {float(r['levels'][-1]):.0f} mmHg, GOA {A / 100:.3f} cm²", fontsize=9)
        ax.set_axis_off()
        print(f"{name}: max von Mises = {r['vm'].max():.3f} MPa at {r['xq'][np.argmax(r['vm'])]}")
        jm, j95, jmed, farmed, nsel = junction_stress(r)
        print(f"{name}: junction band (|x-M|<1.5 mm, 0.2<z<0.8): vM max {jm:.3f}, p95 {j95:.3f}, "
              f"median {jmed:.3f} MPa ({nsel} cells); median away from junction {farmed:.3f} MPa")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fem_goa.pdf"), bbox_inches="tight")
