"""A5: relative comparisons from sim/out/fem/summary.csv (GOA and max|u| at exactly 75 mmHg).
Run: uv run --project sim python sim/scripts/fem_compare.py  -> sim/out/fem/compare.txt
"""
import csv
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "fem")
rows = {r["run"]: r for r in csv.DictReader(open(os.path.join(OUT, "summary.csv")))}
g = lambda n, k="GOA_cm2_at75": float(rows[n][k])
pct = lambda a, b: 100.0 * (a - b) / b
lines = []
for tag in ("_c", "_m", "_f", "_mz4"):
    lines.append(f"fillet vs crease {tag:5s}: GOA {pct(g('fillet' + tag), g('crease' + tag)):+.2f}%  "
                 f"max|u| {pct(g('fillet' + tag, 'max_u_mm_at75'), g('crease' + tag, 'max_u_mm_at75')):+.2f}%")
for geo in ("crease", "fillet"):
    lines.append(f"{geo}: mesh 24x33->40x56 {pct(g(geo + '_m'), g(geo + '_c')):+.2f}%, 40x56->56x78 {pct(g(geo + '_f'), g(geo + '_m')):+.2f}%; "
                 f"corner cut 1/8->1/4 (40x56) {pct(g(geo + '_mz4'), g(geo + '_m')):+.2f}%")
lines.append(f"crease: 2 -> 4 layers (40x56) {pct(g('crease_m4L'), g('crease_m')):+.2f}%")
lines.append("pressure reached: " + ", ".join(f"{n} {float(r['p_reached_mmHg']):.2f}" for n, r in rows.items()))
txt = "\n".join(lines)
print(txt)
open(os.path.join(OUT, "compare.txt"), "w").write(txt + "\n")
