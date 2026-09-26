"""A5 summary: mesh convergence table and GOA(p) curves for all FEM results.
Run: uv run --project sim python sim/scripts/fem_summary.py
Writes sim/out/fem/summary.csv and notes/technote/figures/fem_goa_curve.pdf.
Results are saved exactly at 75 and 80 mmHg (no interpolation), the corner at
the base of post P is removed from the solid and kept as a fixed ghost patch for the GOA, and runs that stop before the
target pressure are reported with the pressure they reached and plotted up to it.
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fem_postprocess import goa, junction_stress  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "fem")
FIG = os.path.join(os.path.dirname(__file__), "..", "..", "notes", "technote", "figures")
RUNS = [(f"{g}{t}", g, m) for g in ("crease", "fillet")
        for t, m in (("_c", "24x33, 2 layers"), ("_m", "40x56, 2 layers"), ("_f", "56x78, 2 layers"),
                     ("_m4L", "40x56, 4 layers"), ("_mz4", "40x56, 2 layers, corner cut z<1/4"))]
BLUE, ORANGE, INK2 = "#2a78d6", "#eb6834", "#52514e"


def at(levels, values, p):
    k = np.where(np.abs(levels - p) < 1e-6)[0]
    return float(values[k[0]]) if len(k) else np.nan


rows = []
fig, ax = plt.subplots(figsize=(4.8, 3.0))
for name, kind, mesh in RUNS:
    f = os.path.join(OUT, f"result_{name}.npz")
    if not os.path.exists(f):
        continue
    r = np.load(f)
    X, tris, R = r["X"], r["tris"], float(r["R"])
    gh = (r["ghostX"], r["ghostT"]) if "ghostX" in r.files else None
    lv = np.asarray(r["levels"], dtype=float)
    G = np.array([goa(X + Ui, tris, R, ghost=gh)[0] / 100 for Ui in r["U"]])
    Mu = np.array([float(np.linalg.norm(Ui, axis=1).max()) for Ui in r["U"]])
    G0 = goa(X, tris, R, ghost=gh)[0] / 100
    pr = float(r["p_reached"])
    row = [name, kind, mesh, len(X), pr, G0]
    for pc in (75.0, 80.0):
        row += [at(lv, G, pc), at(lv, Mu, pc)]
    row += [np.nan] * 5           # 120 mmHg columns are filled below only if EVERY run of this geometry reached 120 mmHg
    rows.append(row)
    if name.endswith(("_m4L", "_mz4")):
        continue                                  # sensitivity runs: in the table, not in the figure
    fine = name.endswith("_f")
    ax.plot(np.concatenate([[0], lv]), np.concatenate([[G0], G]), color=BLUE if kind == "crease" else ORANGE,
            marker="s" if fine else "o", ms=3, lw=1.4 if fine else 0.8, ls="-" if kind == "crease" else "--",
            label=f"{kind} ({mesh})")
ax.axhline(2.32, color=INK2, lw=0.8, ls=":")
ax.text(2, 2.36, "paper, G1 at 120 mmHg (LS-Dyna, with contact)", color=INK2, fontsize=7)
ax.set_ylim(-0.05, 2.6)
ax.set_xlabel("pressure (mmHg)"); ax.set_ylabel("GOA (cm²)")
ax.legend(fontsize=6, frameon=False, loc="lower right")
ax.grid(True, color="#e1e0d9", lw=0.5)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fem_goa_curve.pdf"), bbox_inches="tight")

# report 120 mmHg only if all meshes of the geometry reached exactly 120 mmHg
for kind in ("crease", "fillet"):
    mine = [r_ for r_ in rows if r_[1] == kind]
    if mine and all(abs(r_[4] - 120.0) < 1e-6 for r_ in mine):
        for r_ in mine:
            res = np.load(os.path.join(OUT, f"result_{r_[0]}.npz"))
            lv = np.asarray(res["levels"], dtype=float)
            X, tris, R = res["X"], res["tris"], float(res["R"])
            gh = (res["ghostX"], res["ghostT"]) if "ghostX" in res.files else None
            k = int(np.where(np.abs(lv - 120.0) < 1e-6)[0][0])
            jm, j95, jmed, farmed, _ = junction_stress(res)
            r_[-5:] = [goa(X + res["U"][k], tris, R, ghost=gh)[0] / 100,
                       float(np.linalg.norm(res["U"][k], axis=1).max()), jmed, j95, farmed]

hdr = ["run", "geometry", "mesh", "surface_vertices", "p_reached_mmHg", "GOA_unloaded_cm2",
       "GOA_cm2_at75", "max_u_mm_at75", "GOA_cm2_at80", "max_u_mm_at80",
       "GOA_cm2_at120", "max_u_mm_at120", "junction_vm_median_MPa", "junction_vm_p95_MPa", "far_vm_median_MPa"]
with open(os.path.join(OUT, "summary.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(hdr)
    w.writerows(rows)
for row in rows:
    print(" | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row))
