"""A7: export the G_N package outputs: parameter tables (CSV) and surface meshes (VTK with normals/crease flags, STL).
Reference design: the paper's model, N = 3, R = 10 mm, H = 15 mm.  Criteria: fixed_R (R = 10 mm) and fixed_H (H = 15 mm).
Run: uv run --project sim python sim/scripts/gn_export.py [nz nt]
Output: deliverable/gn/table_fixed_R.csv, table_fixed_H.csv (versioned); sim/out/gn/*.vtk, *.stl (generated).
"""
import csv
import os
import sys

from wheatley import gn

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DELIV = os.path.join(ROOT, "deliverable", "gn")
OUT = os.path.join(ROOT, "sim", "out", "gn")
os.makedirs(DELIV, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
nz, nt = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (40, 60)
S = gn.family(3, "paper", R=10.0, H=15.0).area()
for crit, kw in (("fixed_R", dict(R=10.0)), ("fixed_H", dict(H=15.0))):
    rows = gn.table(criterion=crit, **kw)
    with open(os.path.join(DELIV, f"table_{crit}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "criterion", "R_mm", "H_mm", "h=H/R", "total_area_mm2", "fold_min_deg", "fold_sup_deg"])
        for r in rows:
            w.writerow([r["N"], crit, f"{r['R']:.6f}", f"{r['H']:.6f}", f"{r['h']:.6f}", f"{r['area']:.4f}",
                        f"{r['fold_min_deg']:.3f}", f"{r['fold_sup_deg']:.3f}"])
    for r in rows:
        G = gn.family(r["N"], crit, S=S, **kw)
        m = G.surface_mesh(nz, nt)
        base = os.path.join(OUT, f"G{r['N']}_{crit}")
        m.write(base + ".vtk")
        m.write(base + ".stl")
        print(f"{crit} N={r['N']}: {len(m.points)} points, {len(m.cells_dict['triangle'])} triangles -> {base}.vtk/.stl")
