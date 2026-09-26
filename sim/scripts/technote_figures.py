"""Data and 3D renders for the figures of notes/technote (reproducible).  Run:
    uv run --project sim python sim/scripts/technote_figures.py

The figures are drawn by LaTeX (TikZ/pgfplots, notes/technote/figures/fig_*.tex), so their text uses the fonts and
notation of the note. This script writes what they read:
  - figures/data/*.dat   curves (x y per line; blank line = new curve)
  - figures/data/points.tex   coordinates of labelled points (\\def macros)
  - figures/leaflet3d_{a,b}.png  PyVista renders (orthographic camera, transparent background)
  - figures/data/leaflet3d_labels.tex   label positions on those renders, from the same camera projection
Cross-check: the curvatures of the junction images in the flat patterns must reproduce the closed form of Theorem 3.
"""
from __future__ import annotations

import os

import numpy as np
import pyvista as pv
from scipy.optimize import brentq

from wheatley import develop as dv
from wheatley import geometry as g
from wheatley import nleaflet as nl

pv.OFF_SCREEN = True
FIG = os.path.join(os.path.dirname(__file__), "..", "..", "notes", "technote", "figures")
DATA = os.path.join(FIG, "data")
os.makedirs(DATA, exist_ok=True)
TH = 2 * np.pi / 3
V1, V2 = g.APEX_SURFACE1, g.APEX_SURFACE2
E2 = np.array([-1.0, 0.0])
P2 = np.array([np.cos(np.pi + TH), np.sin(np.pi + TH)])


def write_curves(name, curves):
    with open(os.path.join(DATA, name + ".dat"), "w") as f:
        f.write("x y\n")
        for k, c in enumerate(curves):
            if k:
                f.write("nan nan\n")                  # curve break (pgfplots: unbounded coords=jump)
            for x, y in np.asarray(c)[:, :2]:
                f.write(f"{x:.5f} {y:.5f}\n")


def z_min_of_t(t):
    """Lowest height at which the stay t lies in Omega (t <= t_M(z)); 0 for t <= 5pi/3."""
    if t <= 5 * np.pi / 3:
        return 0.0
    return brentq(lambda z: g.t_M(z) - t, 0.0, 1.0)


def section(z, k=0, n=120, N=3):
    t = np.linspace(np.pi, float(nl.t_M(N, z)), n)
    s1 = g.rot_ccw(g.surface1(t, z), 2 * np.pi * k / N)
    s2 = g.rot_ccw(nl.surface2(N, t, z), 2 * np.pi * k / N)
    return s1, s2


points = {}


def pt(name, p):
    points[name] = np.asarray(p, float)[:2]


# ---------------------------------------------------------------- construction
def construction():
    ring = np.linspace(0, 2 * np.pi, 241)
    write_curves("ring", [np.c_[np.cos(ring), np.sin(ring)]])
    z = 0.6
    a, b = g.radii(z)
    C1, C2 = np.array([-b, 0.0]), g.rot_ccw(np.array([-a, 0.0, 0.0]), TH)[:2]
    write_curves("c_big", [np.c_[C1[0] + a * np.cos(ring), C1[1] + a * np.sin(ring)]])
    write_curves("c_small", [np.c_[C2[0] + b * np.cos(ring), C2[1] + b * np.sin(ring)]])
    s1, s2 = section(z)
    write_curves("c_arc1", [s1]); write_curves("c_arc2", [s2])
    write_curves("c_others", [c for k in (1, 2) for c in section(z, k)])
    for n, p in (("C1", C1), ("C2", C2), ("E", E2), ("P", P2), ("M", g.junction_point(z)), ("O", (0, 0))):
        pt("con" + n, p)
    ang = np.linspace(np.pi, np.pi + TH, 40)
    write_curves("c_theta", [0.2 * np.c_[np.cos(ang), np.sin(ang)]])
    pt("conTh", 0.33 * np.array([np.cos(np.pi + TH / 2), np.sin(np.pi + TH / 2)]))
    zs = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for i, zz in enumerate(zs):
        s1, s2 = section(zz)
        write_curves(f"sec{i}_1", [s1]); write_curves(f"sec{i}_2", [s2])
        pt(f"secM{i}", g.junction_point(zz))
    write_curves("junction_top", [g.junction_point(np.linspace(0, 1, 101))])


# ---------------------------------------------------------------- flat patterns
def inward_curvature(surface, z, h=1e-5):
    """Signed curvature of the junction image in the flat pattern of `surface`, with the normal pointing into the
    pattern (towards smaller t); this is the geodesic curvature k_i of Theorem 3."""
    t = g.t_M(z)
    J = lambda zz: dv.develop(surface, g.t_M(zz), zz)
    d1 = (J(z + h) - J(z - h)) / (2 * h)
    d2 = (J(z + h) - 2 * J(z) + J(z - h)) / h**2
    inward = -(dv.develop(surface, t + h, z) - dv.develop(surface, t - h, z)) / (2 * h)
    speed = np.hypot(d1[..., 0], d1[..., 1])
    nrm = np.stack([-d1[..., 1], d1[..., 0]], -1) / speed[..., None]
    sgn = np.sign(np.sum(nrm * inward, -1))
    return sgn * (d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]) / speed**3


def flat_patterns():
    for s in (1, 2):
        zlines = []
        for z in np.linspace(0, 1, 6)[1:-1]:
            t = np.linspace(np.pi, float(g.t_M(z)), 100)
            zlines.append(dv.develop(s, t, np.full_like(t, z)))
        write_curves(f"flat{s}_z", zlines)
        for tag, z in (("base", 0.0), ("top", 1.0)):
            if s == 2 and z == 0.0:
                continue
            t = np.linspace(np.pi, float(g.t_M(z)), 120)
            write_curves(f"flat{s}_{tag}", [dv.develop(s, t, np.full_like(t, z))])
        stays = []
        for t in np.linspace(np.pi, 2 * np.pi, 11)[:-1]:
            zz = np.linspace(z_min_of_t(t), 1, 30)
            stays.append(dv.develop(s, np.full_like(zz, t), zz))
        write_curves(f"flat{s}_stays", stays)
        write_curves(f"flat{s}_junction", [dv.junction_image(s, np.linspace(0, 1, 121))])
        tm = 0.5 * (np.pi + float(g.t_M(1.0)))
        pt(f"flat{s}top", dv.develop(s, tm, 1.0))
        pt(f"flat{s}J", dv.junction_image(s, 0.5))
        if s == 1:
            pt("flatonebase", dv.develop(1, 0.5 * (np.pi + float(g.t_M(0.0))), 0.0))
    z = np.linspace(0.02, 0.98, 97)
    k1, k2 = inward_curvature(1, z), inward_curvature(2, z)
    Dz = z**2 - 2 * z + 4
    closed = 4 * np.sqrt(3) * (z - 1) * Dz**2 / (np.sqrt(Dz**2 + 9) * (Dz**2 + 12) ** 1.5)
    err = float(np.max(np.abs(k1 + k2 - closed)))
    assert err < 1e-4, f"flat-pattern curvatures disagree with Theorem 3: {err}"
    write_curves("kg1", [np.c_[z, k1]]); write_curves("kg2neg", [np.c_[z, -k2]])
    print(f"flat patterns: max |k1 + k2 - closed form of Theorem 3| = {err:.1e}")


# ---------------------------------------------------------------- fold angle, top views
def fold_and_top():
    z = np.linspace(1e-3, 1, 200)
    for N in range(2, 8):
        write_curves(f"fold{N}", [np.c_[z, nl.fold_angle_deg(N, z)]])
    b = z / 2
    D = 1 - b + b * b
    h = 1.5
    write_curves("fold3_h15", [np.c_[z, np.degrees(np.arccos((8 * h * h * D * D + 9) / (16 * h * h * D * D + 9)))]])
    for N in (2, 3, 4, 6):
        for tag, zs in (("sec", (0.25, 0.5, 0.75)), ("edge", (1.0,))):
            c1, c2 = [], []
            for k in range(N):
                for zz in zs:
                    s1, s2 = section(zz, k, n=100, N=N)
                    c1.append(s1); c2.append(s2)
            write_curves(f"top{N}_{tag}1", c1); write_curves(f"top{N}_{tag}2", c2)


# ---------------------------------------------------------------- 3D renders (PyVista)
BLUE_S, ORANGE_S, INK = "#8fb8ea", "#f2ad8a", "#161616"
BLUE_D, ORANGE_D = "#2a78d6", "#e0662c"


def grid_mesh(P):
    return pv.StructuredGrid(P[..., 0], P[..., 1], P[..., 2])


def tube(pts, r):
    return pv.lines_from_points(np.asarray(pts)).tube(radius=r, n_sides=16)


def dashed(p, q, r, n=9):
    segs = []
    for i in range(n):
        a, b = p + (q - p) * (2 * i) / (2 * n - 1), p + (q - p) * (2 * i + 1) / (2 * n - 1)
        segs.append(tube([a, b], r))
    return segs


def render(name, actors, focal, elev, azim, scale, labels, size=(1500, 1200)):
    """Orthographic render; returns label positions normalised to the cropped image (0..1 from bottom left)."""
    pl = pv.Plotter(window_size=size, lighting="none")
    pl.add_light(pv.Light(light_type="headlight", intensity=0.42))
    pl.add_light(pv.Light(position=(2.5, -3.0, 4.0), focal_point=(0, 0, 0.5), light_type="scene light", intensity=0.42))
    for mesh, kw in actors:
        pl.add_mesh(mesh, **kw)
    pl.enable_parallel_projection()
    e, az = np.radians(elev), np.radians(azim)
    d = np.array([np.cos(e) * np.cos(az), np.cos(e) * np.sin(az), np.sin(e)])       # from focal point to camera
    focal = np.asarray(focal, float)
    pl.camera.focal_point = focal
    pl.camera.position = focal + 10 * d
    pl.camera.up = (0, 0, 1)
    pl.camera.parallel_scale = scale
    pl.enable_anti_aliasing("ssaa")
    img = pl.screenshot(os.path.join(FIG, name + ".png"), transparent_background=True, scale=2)
    H, W = img.shape[:2]
    view = -d
    right = np.cross(view, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, view)
    alpha = img[..., 3] > 0
    ys, xs = np.where(alpha)
    pad = 24                                                   # keep markers at the border (apex spheres) whole
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + 1 + pad, H)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + 1 + pad, W)
    from PIL import Image
    Image.fromarray(img[y0:y1, x0:x1]).save(os.path.join(FIG, name + ".png"))
    out = {}
    for key, p in labels.items():
        u = W / 2 + np.dot(np.asarray(p) - focal, right) / scale * (H / 2)
        v = H / 2 - np.dot(np.asarray(p) - focal, up) / scale * (H / 2)
        out[key] = ((u - x0) / (x1 - x0), 1 - (v - y0) / (y1 - y0))
    pl.close()
    return out, (x1 - x0) / (y1 - y0)


def leaflet3d():
    surf = dict(smooth_shading=True, specular=0.08, specular_power=12, ambient=0.32, diffuse=0.7)
    P1, P2 = g.leaflet_grid(nt=90, nz=70, z_min=1e-3)
    M = g.junction_point(np.linspace(0, 1, 150))
    ring = np.linspace(0, 2 * np.pi, 200)
    ring3 = np.c_[np.cos(ring), np.sin(ring), 0 * ring]
    acts = [(grid_mesh(P1), dict(color=BLUE_S, **surf)), (grid_mesh(P2), dict(color=ORANGE_S, **surf)),
            (tube(ring3, 0.006), dict(color="#8a8984")), (tube(M, 0.014), dict(color=INK))]
    for t in np.linspace(np.pi + 0.22, 5 * np.pi / 3 - 0.06, 4):
        L = g.surface1(np.full(2, t), np.array([0.0, 1.0]))
        acts.append((tube(L, 0.008), dict(color=BLUE_D)))
        acts += [(s, dict(color=BLUE_D)) for s in dashed(L[1], V1, 0.005)]
        L2 = g.surface2(np.full(2, t), np.array([0.0, 1.0]))
        acts.append((tube(L2, 0.008), dict(color=ORANGE_D)))
    for p in (V1, V2):
        acts.append((pv.Sphere(radius=0.03, center=p), dict(color=INK, smooth_shading=True)))
    Mhalf = g.junction_point(0.95)                             # label the part of the junction that is visible
    lab_a, ar_a = render("leaflet3d_a", acts, focal=(0, -0.2, 1.02), elev=16, azim=-62, scale=1.3,
                         labels={"Vone": V1, "Vtwo": V2, "M": Mhalf, "E": (-1, 0, 0)})
    acts = []
    for k in range(3):
        Q1, Q2 = g.leaflet_grid(nt=70, nz=55, k=k, z_min=1e-3)
        # render only: pull each leaflet 0.4% towards its own centroid so that the surfaces of neighbouring leaflets,
        # which coincide along the posts and the free edges, do not z-fight
        c = np.r_[np.concatenate([Q1, Q2]).reshape(-1, 3)[:, :2].mean(0), 0.0]
        Q1, Q2 = c + 0.996 * (Q1 - c), c + 0.996 * (Q2 - c)
        acts += [(grid_mesh(Q1), dict(color=BLUE_S, **surf)), (grid_mesh(Q2), dict(color=ORANGE_S, **surf)),
                 (tube(g.rot_ccw(M, k * TH), 0.012), dict(color=INK))]
    acts.append((tube(ring3, 0.006), dict(color="#8a8984")))
    lab_b, ar_b = render("leaflet3d_b", acts, focal=(0, 0, 0.4), elev=42, azim=-60, scale=0.95, labels={"O": (0, 0, 1)})
    with open(os.path.join(DATA, "leaflet3d_labels.tex"), "w") as f:
        f.write("% generated by sim/scripts/technote_figures.py: label positions on the renders (fractions)\n")
        for pre, lab in (("A", lab_a), ("B", lab_b)):
            for k, (u, v) in lab.items():
                f.write(f"\\def\\lab{pre}{k}{{{u:.4f},{v:.4f}}}\n")
        f.write(f"\\def\\aspectA{{{ar_a:.4f}}}\n\\def\\aspectB{{{ar_b:.4f}}}\n")


def write_points():
    with open(os.path.join(DATA, "points.tex"), "w") as f:
        f.write("% generated by sim/scripts/technote_figures.py\n")
        for k, (x, y) in points.items():
            f.write(f"\\def\\pt{k.replace('0','zero').replace('1','one').replace('2','two').replace('3','three').replace('4','four').replace('5','five')}{{({x:.5f},{y:.5f})}}\n")


if __name__ == "__main__":
    construction(); flat_patterns(); fold_and_top(); write_points(); leaflet3d()
    print("written to", os.path.abspath(FIG))
