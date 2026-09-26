"""A5: quasi-static opening of ONE leaflet under uniform follower pressure (thin 3D solid, P2 tets).

Model (see CLAIMS.md C-020 for status and caveats):
  - geometry: sim/out/fem/leaflet_{crease,fillet}.npz (sim/scripts/make_leaflet_meshes.py), mm;
  - material: St Venant-Kirchhoff, E = 40 MPa, nu = 0.49 (paper Table 1), thickness 0.3 mm;
  - clamping: all translations fixed on the base arc and on both commissure posts (paper p. 11-12);
  - load: uniform follower pressure on the ventricular (inner) face, 0 -> p_max (default 120 mmHg);
  - no contact (leaflets move apart when opening), no inertia, no stays (design G1).
Run (Docker, PETSc):
  tools/fenicsx-docker.sh mpirun -n 12 python3 sim/fem/leaflet_pressure.py crease
Output: sim/out/fem/result_<name>.npz (midsurface displacement at each saved load level, summary).
"""
import sys
import time

import basix.ufl
import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from scipy.spatial import cKDTree

comm = MPI.COMM_WORLD
name = sys.argv[1] if len(sys.argv) > 1 else "crease"
p_max_mmHg = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
MMHG = 1.33322e-4          # MPa
E_mod, nu = 40.0, 0.49     # MPa

d = np.load(f"sim/out/fem/leaflet_{name}.npz")
Y, tets, X, nrm = d["Y"], d["tets"], d["X"], d["n"]
T, R, H = float(d["T"]), float(d["R"]), float(d["H"])
Epost, Ppost = d["E"], d["P"]

cel = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
if comm.rank == 0:
    msh = mesh.create_mesh(comm, tets.astype(np.int64), cel, Y)
else:
    msh = mesh.create_mesh(comm, np.empty((0, 4), dtype=np.int64), cel, np.empty((0, 3)))
tdim = msh.topology.dim

V = fem.functionspace(msh, ("Lagrange", 2, (3,)))
u = fem.Function(V, name="u")
v = ufl.TestFunction(V)

# ---- clamping: base arc (z = 0 ring) and the two vertical posts (through E and P)
tree = cKDTree(X)


def clamp(x):
    xy = np.sqrt(x[0] ** 2 + x[1] ** 2)
    base = np.sqrt((xy - R) ** 2 + x[2] ** 2) < 0.6 * T
    postE = np.hypot(x[0] - Epost[0], x[1] - Epost[1]) < 0.6 * T
    postP = np.hypot(x[0] - Ppost[0], x[1] - Ppost[1]) < 0.6 * T
    return base | postE | postP


dofs = fem.locate_dofs_geometrical(V, clamp)
bc = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), dofs, V)

# ---- inner (ventricular) face: all vertices on the -T/2 offset
def inner(x):
    _, i = tree.query(x.T)
    s = np.einsum("ij,ij->i", x.T - X[i], nrm[i])
    return s < -0.3 * T


msh.topology.create_connectivity(tdim - 1, tdim)
bfacets = mesh.locate_entities_boundary(msh, tdim - 1, inner)
ft = mesh.meshtags(msh, tdim - 1, np.sort(bfacets), np.full(len(bfacets), 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)

# ---- St Venant-Kirchhoff
lmbda = E_mod * nu / ((1 + nu) * (1 - 2 * nu))
mu = E_mod / (2 * (1 + nu))
I = ufl.Identity(3)
F = I + ufl.grad(u)
Eg = 0.5 * (F.T * F - I)
S = lmbda * ufl.tr(Eg) * I + 2 * mu * Eg
P = F * S
J = ufl.det(F)
N0 = ufl.FacetNormal(msh)
p = fem.Constant(msh, default_scalar_type(0.0))
Res = ufl.inner(P, ufl.grad(v)) * ufl.dx + p * J * ufl.dot(ufl.inv(F).T * N0, v) * ds(1)

import os
DEBUG = os.environ.get("LEAF_DEBUG") == "1"
ndofs_bc = comm.allreduce(len(dofs), op=MPI.SUM)
nfac = comm.allreduce(len(bfacets), op=MPI.SUM)
if comm.rank == 0:
    print(f"clamped dofs (blocks): {ndofs_bc}; inner facets: {nfac}; V dofs: "
          f"{V.dofmap.index_map.size_global * 3}", flush=True)
opts_dbg = {"snes_monitor": None, "ksp_converged_reason": None, "snes_converged_reason": None,
            "snes_max_it": 6, "mat_mumps_icntl_14": 200} if DEBUG else {}
problem = NonlinearProblem(
    Res, u, bcs=[bc], petsc_options_prefix="leaf_",
    petsc_options={"snes_type": os.environ.get("LEAF_SNES", "newtonls"),
                   "snes_linesearch_type": os.environ.get("LEAF_LS", "basic"), "snes_rtol": 1e-8,
                   "snes_atol": 1e-10, "snes_stol": 0.0, "snes_max_it": 40, "ksp_type": "preonly",
                   "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", **opts_dbg})

# ---- gather midsurface displacement (P1 interpolation at mesh vertices)
V1 = fem.functionspace(msh, ("Lagrange", 1, (3,)))
u1 = fem.Function(V1)


def midsurface_u():
    u1.interpolate(u)
    nloc = V1.dofmap.index_map.size_local
    xs = V1.tabulate_dof_coordinates()[:nloc]
    us = u1.x.array.reshape(-1, 3)[:nloc]
    xs = comm.gather(xs, root=0)
    us = comm.gather(us, root=0)
    if comm.rank == 0:
        xs, us = np.vstack(xs), np.vstack(us)
        dist, i = cKDTree(xs).query(X)
        assert dist.max() < 1e-8, dist.max()
        return us[i]
    return None


def vm_max():
    sig = (1 / J) * F * S * F.T
    dev = sig - ufl.tr(sig) / 3 * I
    vm = ufl.sqrt(1.5 * ufl.inner(dev, dev))
    Q = fem.functionspace(msh, ("DG", 0))
    q = fem.Function(Q)
    q.interpolate(fem.Expression(vm, Q.element.interpolation_points))
    nloc = Q.dofmap.index_map.size_local
    xq = Q.tabulate_dof_coordinates()[:nloc]
    vals = q.x.array[:nloc]
    allx = comm.gather(xq, root=0)
    allv = comm.gather(vals, root=0)
    if comm.rank == 0:
        return np.vstack(allx), np.concatenate(allv)
    return None, None


t0 = time.time()
levels, U, stats = [], [], []
target = p_max_mmHg * MMHG
pc, dp, dp_min = 0.0, target / 40, target / 5000
u_prev = u.x.array.copy()
# exact save levels (no interpolation between saved levels): every 15 mmHg plus 75 and 80 mmHg
save_levels = sorted({round(v, 9) for v in list(np.arange(15.0, p_max_mmHg + 1e-9, 15.0)) + [75.0, 80.0, p_max_mmHg]
                      if v <= p_max_mmHg + 1e-9})
save_levels = [v * MMHG for v in save_levels]
hist = []                         # (p, its, max|u|) of every converged step
last_saved = -1.0                 # pressure of the last saved state, known on EVERY rank (MPI)
if DEBUG:
    target = dp
while pc < target - 1e-15:
    nxt = next((v for v in save_levels if v > pc + 1e-12), target)
    step = min(dp, target - pc, nxt - pc)
    p.value = pc + step
    try:
        problem.solve()
        reason = problem.solver.getConvergedReason()
        its = problem.solver.getIterationNumber()
    except Exception as e:           # noqa: BLE001
        reason, its = -99, -1
    if reason > 0:
        pc += step
        u_prev[:] = u.x.array
        umax = comm.allreduce(np.max(np.linalg.norm(u.x.array.reshape(-1, 3), axis=1))
                              if len(u.x.array) else 0.0, op=MPI.MAX)
        if comm.rank == 0:
            print(f"p = {pc / MMHG:8.3f} mmHg  its {its:2d}  max|u| = {umax:.4f} mm  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        hist.append((pc / MMHG, its, umax))
        if its <= 6:
            dp = min(dp * 1.5, target / 10)
        if any(abs(pc - v) < 1e-9 * MMHG for v in save_levels):
            Um = midsurface_u()
            last_saved = pc
            if comm.rank == 0:
                levels.append(pc / MMHG)
                U.append(Um)
    else:
        u.x.array[:] = u_prev
        u.x.scatter_forward()
        dp = step / 2
        if comm.rank == 0:
            print(f"  step {step / MMHG:.4f} mmHg failed (reason {reason}); retry with {dp / MMHG:.4f}",
                  flush=True)
        if dp < dp_min:
            if comm.rank == 0:
                print(f"  giving up: step below minimum {dp_min / MMHG:.4f} mmHg (last attempted {step / MMHG:.4f})",
                      flush=True)
            if abs(last_saved - pc) > 1e-9 * MMHG:                   # keep the last converged state (collective)
                u.x.array[:] = u_prev
                u.x.scatter_forward()
                Um = midsurface_u()
                if comm.rank == 0:
                    levels.append(pc / MMHG)
                    U.append(Um)
            break

xq, vm = vm_max()
if comm.rank == 0:
    extra = {k: d[k] for k in ("ghostX", "ghostT", "corner_zc") if k in d.files}
    np.savez(f"sim/out/fem/result_{name}.npz", levels=np.array(levels), U=np.array(U), X=X,
             tris=d["tris"], vm=vm, xq=xq, p_reached=pc / MMHG, R=R, H=H, T=T, hist=np.array(hist), **extra)
    Uf = U[-1] if U else np.zeros_like(X)
    mag = np.linalg.norm(Uf, axis=1)
    print(f"DONE {name}: p = {pc / MMHG:.2f} mmHg; max|u| = {mag.max():.3f} mm; "
          f"u_z in [{Uf[:, 2].min():.3f}, {Uf[:, 2].max():.3f}] mm; max von Mises = {vm.max():.3f} MPa; "
          f"time {time.time() - t0:.0f}s", flush=True)
