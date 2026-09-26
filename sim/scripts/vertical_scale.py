"""CONVENTIONS A-13: effect of a vertical scale h (X_h = (x, y, h z)) on the crease angle and area density.
h = 1 is the paper's normalised family; h = 1.5 corresponds to 15 mm height on a 10 mm radius IF the CAD used a
pure vertical stretch (not confirmed)."""
import numpy as np
from wheatley import geometry as g

def scaled(f, h):
    return lambda t, z: f(t, z) * np.array([1.0, 1.0, h])

def crease(z, h, eps=1e-6):
    n = []
    for f in (g.surface1, g.surface2):
        F = scaled(f, h); t = g.t_M(z)
        Xt = (F(t + eps, z) - F(t - eps, z)) / (2 * eps); Xz = (F(t, z + eps) - F(t, z - eps)) / (2 * eps)
        v = np.cross(Xt, Xz); n.append(v / np.linalg.norm(v))
    return np.degrees(np.arccos(abs(n[0] @ n[1])))

def area_density(f, t, z, h, eps=1e-6):
    F = scaled(f, h)
    Xt = (F(t + eps, z) - F(t - eps, z)) / (2 * eps); Xz = (F(t, z + eps) - F(t, z - eps)) / (2 * eps)
    return np.linalg.norm(np.cross(Xt, Xz))

for h in (1.0, 1.5):
    print(f"h={h}: crease(z=1)={crease(1 - 1e-9, h):.6f} deg, formula acos((h^2+2)/(2(h^2+1)))="
          f"{np.degrees(np.arccos((h*h+2)/(2*(h*h+1)))):.6f}; |X_t x X_z|(t=pi, z=1/2) = {area_density(g.surface1, np.pi, 0.5, h):.6f}")
