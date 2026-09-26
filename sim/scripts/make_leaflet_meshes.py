"""Build thin-solid meshes of one leaflet (N = 3) for the A5 comparison: sharp crease vs G1 fillet.
Physical units mm; ring radius 10 mm; height scale h = 1.5 (15 mm, paper Fig. 8; A-13 unconfirmed);
thickness 0.3 mm (paper p. 14).  Output: sim/out/fem/leaflet_{crease,fillet}.npz
The corner at the base of post P (z < CORNER_ZC = 1/8 by default; nz must be a multiple of 8) is
removed identically in both geometries and kept as a fixed 'ghost' patch for the GOA; the fillet radius is
sqrt((rho0 z)^2 + (T/R)^2) (smooth, so the rounded leaflet stays G1); `thin_solid` raises if any prism is twisted or
any two tetrahedra overlap (exact global separating-axis check).
Run: uv run --project sim python sim/scripts/make_leaflet_meshes.py [nz n1 nf n2 layers]
"""
import os
import sys

import numpy as np

from wheatley import leaflet_mesh as lm

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "fem")
os.makedirs(OUT, exist_ok=True)
nz, n1, nf, n2, L = (int(a) for a in (sys.argv[1:6] if len(sys.argv) > 5 else (40, 56, 8, 34, 2)))
R, H, T = 10.0, 1.5, 0.3

TAG = os.environ.get("MESH_TAG", "")
ZC = float(os.environ.get("CORNER_ZC", "0.125"))
for name, rho0 in (("crease" + TAG, 0.0), ("fillet" + TAG, 0.10)):
    G = lm.surface_grid(nz, n1, n2, nf, rho0=rho0, radius=R, h=H, rho_min=T / R)
    skip, kcut = lm.corner_skip(G, n1, ZC)
    X, tris, idx = lm.triangulate(G, skip=skip)
    Y, tets, n, layer = lm.thin_solid(X, tris, thickness=T, layers=L)       # raises if the solid is invalid
    gX, gT, _ = lm.triangulate(G, skip=~skip)                                # the removed corner (for the GOA only)
    # quality: min dihedral-free proxy = volume / (edge^3)
    p = Y[tets]
    vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6
    edges = [np.linalg.norm(p[:, i] - p[:, j], axis=1) for i in range(4) for j in range(i + 1, 4)]
    lmax = np.max(edges, axis=0)
    q = vol / lmax**3
    np.savez(os.path.join(OUT, f"leaflet_{name}.npz"), Y=Y, tets=tets, X=X, tris=tris, n=n,
             layer=layer, nv=len(X), R=R, H=H, T=T, L=L,
             E=np.array(G[0, 0]), P=np.array(G[0, -1]), ghostX=gX, ghostT=gT, corner_zc=ZC)
    print(f"{name}: corner rows removed {kcut} (z < {ZC:g}); surface verts {len(X)}, tris {len(tris)}, tets {len(tets)}, "
          f"min vol {vol.min():.2e}, quality min {q.min():.2e} median {np.median(q):.2e}")
