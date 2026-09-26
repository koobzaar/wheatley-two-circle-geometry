"""A5: is the stop of the quasi-static Newton near ~85 mmHg a limit point of the model or a numerical failure?

Same model as sim/fem/leaflet_pressure.py (copied set-up; that script is left untouched).  Stages:
  1. march in pressure with full Newton steps ("basic"), keeping the FULL P2 field of the last converged states in memory
     (checkpoints), until the step falls below the minimum;
  2. restart from the SAME last converged state with a line search ("bt") and with a trust region ("newtontr");
  3. at the last converged states: min det(F) at quadrature points; the smallest-magnitude eigenvalues and the smallest
     singular value of the tangent K = dR/du restricted to the FREE dofs (Dirichlet rows/columns removed, not replaced by 1);
  4. pseudo-arclength continuation from the last converged state: residual R(u, p) = F_int(u) + p g(u) (follower pressure,
     g = dR/dp assembled at the current u), constraint N = tau_u . (u - u_n) + alpha^2 tau_p (p - p_n) - ds = 0 with the
     previous unit tangent tau (Euclidean metric on free dofs; alpha fixed per run, alpha = ||K0^-1 g0|| at the start of
     the continuation), bordering: K a = -R, K b = -g, dp = (-N - tau_u.a) / (tau_u.b + alpha^2 tau_p), du = a + b dp;
     tangent K t_u = -g t_p, oriented to keep a positive product with the previous tangent (so p may decrease).
Output: sim/out/fem/limit_<name>.npz and a log on stdout.
Run: tools/fenicsx-docker.sh mpirun -n 12 python3 sim/fem/leaflet_limit.py <mesh name> [p_max_mmHg] [n_arc_steps]
"""
import os
import sys
import time

import basix.ufl
import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem.petsc import (NonlinearProblem, apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, set_bc)
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree

comm = MPI.COMM_WORLD
name = sys.argv[1] if len(sys.argv) > 1 else "crease_c"
p_max_mmHg = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
n_arc = int(sys.argv[3]) if len(sys.argv) > 3 else 40
MMHG = 1.33322e-4
E_mod, nu = 40.0, 0.49
log = (lambda *a: print(*a, flush=True)) if comm.rank == 0 else (lambda *a: None)

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
du_trial = ufl.TrialFunction(V)
tree = cKDTree(X)


def clamp(x):
    xy = np.sqrt(x[0] ** 2 + x[1] ** 2)
    base = np.sqrt((xy - R) ** 2 + x[2] ** 2) < 0.6 * T
    postE = np.hypot(x[0] - Epost[0], x[1] - Epost[1]) < 0.6 * T
    postP = np.hypot(x[0] - Ppost[0], x[1] - Ppost[1]) < 0.6 * T
    return base | postE | postP


dofs = fem.locate_dofs_geometrical(V, clamp)
bc = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), dofs, V)


def inner(x):
    _, i = tree.query(x.T)
    return np.einsum("ij,ij->i", x.T - X[i], nrm[i]) < -0.3 * T


msh.topology.create_connectivity(tdim - 1, tdim)
bfacets = mesh.locate_entities_boundary(msh, tdim - 1, inner)
ft = mesh.meshtags(msh, tdim - 1, np.sort(bfacets), np.full(len(bfacets), 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
lmbda = E_mod * nu / ((1 + nu) * (1 - 2 * nu))
mu = E_mod / (2 * (1 + nu))
I = ufl.Identity(3)
F = I + ufl.grad(u)
Eg = 0.5 * (F.T * F - I)
S = lmbda * ufl.tr(Eg) * I + 2 * mu * Eg
J = ufl.det(F)
N0 = ufl.FacetNormal(msh)
p = fem.Constant(msh, default_scalar_type(0.0))
gvec_form = J * ufl.dot(ufl.inv(F).T * N0, v) * ds(1)            # dR/dp
Res = ufl.inner(F * S, ufl.grad(v)) * ufl.dx + p * gvec_form
Jac = ufl.derivative(Res, u, du_trial)
Res_c, Jac_c, g_c = fem.form(Res), fem.form(Jac), fem.form(gvec_form)


def make_problem(snes, ls):
    return NonlinearProblem(Res, u, bcs=[bc], petsc_options_prefix=f"lim_{snes}_{ls}_",
                            petsc_options={"snes_type": snes, "snes_linesearch_type": ls, "snes_rtol": 1e-8,
                                           "snes_atol": 1e-10, "snes_stol": 0.0, "snes_max_it": 40,
                                           "ksp_type": "preonly", "pc_type": "lu",
                                           "pc_factor_mat_solver_type": "mumps"})


def solve_at(problem, pval):
    p.value = pval
    try:
        problem.solve()
        return problem.solver.getConvergedReason(), problem.solver.getIterationNumber()
    except Exception:            # noqa: BLE001
        return -99, -1


# ---- free-dof index set (Dirichlet rows/columns removed)
bs = V.dofmap.index_map_bs
nloc = V.dofmap.index_map.size_local * bs
rstart = V.dofmap.index_map.local_range[0] * bs
bc_local = np.asarray(bc.dof_indices()[0])
bc_local = bc_local[bc_local < nloc]
free_local = np.setdiff1d(np.arange(nloc), bc_local)
IS_free = PETSc.IS().createGeneral((rstart + free_local).astype(PETSc.IntType), comm=comm)


def assemble_K():
    A = assemble_matrix(Jac_c, bcs=[bc])
    A.assemble()
    return A


def assemble_g():
    b = create_vector(g_c)
    with b.localForm() as bl:
        bl.set(0.0)
    assemble_vector(b, g_c)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc], alpha=0.0)
    return b


def assemble_R():
    b = create_vector(Res_c)
    with b.localForm() as bl:
        bl.set(0.0)
    assemble_vector(b, Res_c)
    apply_lifting(b, [Jac_c], bcs=[[bc]], x0=[u.x.petsc_vec], alpha=-1.0)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc], u.x.petsc_vec, -1.0)
    return b


def lu_solver(A):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    ksp.setFromOptions()
    return ksp


def min_detF():
    Q = fem.functionspace(msh, ("DG", 2))               # det F is a polynomial of degree <= 3 per cell for P2
    q = fem.Function(Q)
    q.interpolate(fem.Expression(J, Q.element.interpolation_points))
    loc = q.x.array[:Q.dofmap.index_map.size_local].min() if Q.dofmap.index_map.size_local else np.inf
    return comm.allreduce(loc, op=MPI.MIN)


def spectrum(nev=6):
    """Smallest-magnitude eigenvalues (non-Hermitian, shift-and-invert near 0) and smallest singular value of K_ff."""
    from slepc4py import SLEPc
    K = assemble_K()
    Kff = K.createSubMatrix(IS_free, IS_free)
    out = {}
    eps = SLEPc.EPS().create(comm)
    eps.setOperators(Kff)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    eps.setDimensions(nev)
    eps.setTarget(-1e-6)                                  # small shift, not exactly 0
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    st = eps.getST()
    st.setType("sinvert")
    ksp = st.getKSP()
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    eps.setFromOptions()
    eps.solve()
    vals = []
    for i in range(eps.getConverged()):
        lam = eps.getEigenvalue(i)
        err = eps.computeError(i)
        vals.append((complex(lam), float(err)))
    out["eig"] = sorted(vals, key=lambda t: abs(t[0]))[:nev]
    if os.environ.get("LIMIT_SVD", "0") != "1":          # the cyclic SVD factorises a 2n x 2n matrix: memory-heavy, opt-in
        out["smin"] = float("nan")
        return out
    svd = SLEPc.SVD().create(comm)
    svd.setOperator(Kff)
    svd.setDimensions(1)
    svd.setWhichSingularTriplets(SLEPc.SVD.Which.SMALLEST)
    # cyclic formulation [0 K; K^T 0] with shift-and-invert near 0: plain Lanczos for the smallest sigma is far too slow
    svd.setType(SLEPc.SVD.Type.CYCLIC)
    svd.setCyclicExplicitMatrix(True)
    ceps = svd.getCyclicEPS()
    ceps.setTarget(1e-8)
    ceps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    cst = ceps.getST()
    cst.setType("sinvert")
    cksp = cst.getKSP()
    cksp.setType("preonly")
    cksp.getPC().setType("lu")
    cksp.getPC().setFactorSolverType("mumps")
    svd.setFromOptions()
    try:
        svd.solve()
        out["smin"] = float(svd.getSingularTriplet(0)) if svd.getConverged() else float("nan")
    except Exception:         # noqa: BLE001
        out["smin"] = float("nan")
    return out


# ---------------------------------------------------------------- stage 1: pressure march with checkpoints
t0 = time.time()
basic = make_problem("newtonls", "basic")
target = p_max_mmHg * MMHG
pc, dp, dp_min = 0.0, target / 40, target / 5000
ckpt = []                                              # (p, full local P2 array) of the last converged states
hist = []
while pc < target - 1e-15:
    step = min(dp, target - pc)
    reason, its = solve_at(basic, pc + step)
    if reason > 0:
        pc += step
        ckpt.append((pc, u.x.array.copy()))
        ckpt = ckpt[-6:]
        hist.append((pc / MMHG, its, reason))
        log(f"[march] p = {pc / MMHG:8.3f} mmHg its {its:2d} ({time.time() - t0:.0f}s)")
        if its <= 6:
            dp = min(dp * 1.5, target / 10)
    else:
        u.x.array[:] = ckpt[-1][1] if ckpt else 0.0
        u.x.scatter_forward()
        hist.append((pc / MMHG + step / MMHG, its, reason))
        dp = step / 2
        log(f"[march]   step {step / MMHG:.4f} failed (reason {reason})")
        if dp < dp_min:
            break
p_star = pc
log(f"[march] stopped at p* = {p_star / MMHG:.4f} mmHg (target {p_max_mmHg})")

# ---------------------------------------------------------------- stage 2: restart from the same state
restart = []
for snes, ls in (("newtonls", "bt"), ("newtontr", "basic")):
    prob = make_problem(snes, ls)
    for frac in (0.25, 1.0, 2.5):
        u.x.array[:] = ckpt[-1][1]
        u.x.scatter_forward()
        stepp = frac * dp_min
        reason, its = solve_at(prob, p_star + stepp)
        restart.append((snes, ls, (p_star + stepp) / MMHG, reason, its))
        log(f"[restart] {snes}/{ls}: p = {(p_star + stepp) / MMHG:.4f} reason {reason} its {its}")
u.x.array[:] = ckpt[-1][1]
u.x.scatter_forward()
p.value = p_star

# ---------------------------------------------------------------- stage 3: detF and spectrum along the last states
diag = []
for pk, arr in ckpt:
    u.x.array[:] = arr
    u.x.scatter_forward()
    p.value = pk
    sp = spectrum()
    dF = min_detF()
    diag.append((pk / MMHG, dF, sp["smin"], [e[0] for e in sp["eig"]], [e[1] for e in sp["eig"]]))
    log(f"[diag] p = {pk / MMHG:8.3f}: min detF = {dF:.4f}; smallest sing. value {sp['smin']:.4e}; "
        f"eig |.| = {[f'{abs(e[0]):.3e}' for e in sp['eig']]}")

# ---------------------------------------------------------------- stage 4: pseudo-arclength continuation
u.x.array[:] = ckpt[-1][1]
u.x.scatter_forward()
p.value = p_star
K = assemble_K()
ksp = lu_solver(K)
g = assemble_g()
b = g.duplicate()
ksp.solve(g, b)
b.scale(-1.0)                                          # K b = -g : du/dp along the equilibrium path
bnorm = b.norm()
alpha = bnorm                                          # alpha = ||K0^-1 g0||: balances the two terms (fixed)
tau_u = b.copy()
tau_p = 1.0
nt = np.sqrt(tau_u.dot(tau_u) + alpha ** 2 * tau_p ** 2)
tau_u.scale(1 / nt)
tau_p /= nt
ds_arc = 0.05 * alpha * (0.5 * MMHG)                   # initial arclength ~ a 0.5 mmHg pressure step
arc = []
un = u.x.petsc_vec.copy()
pn = p_star
for k in range(n_arc):
    ok = False
    for attempt in range(6):
        # predictor
        u.x.petsc_vec.waxpy(ds_arc, tau_u, un)
        u.x.scatter_forward()
        pval = pn + ds_arc * tau_p
        for it in range(25):
            p.value = pval
            Rv = assemble_R()
            du_ = u.x.petsc_vec.copy()
            du_.axpy(-1.0, un)
            Nres = tau_u.dot(du_) + alpha ** 2 * tau_p * (pval - pn) - ds_arc
            rn = Rv.norm()
            if rn < 1e-8 * max(1.0, g.norm() * abs(pval)) and abs(Nres) < 1e-10 * max(ds_arc, 1e-30):
                ok = True
                break
            Kk = assemble_K()
            kk = lu_solver(Kk)
            a = Rv.duplicate()
            Rv.scale(-1.0)
            kk.solve(Rv, a)                            # K a = -R
            gk = assemble_g()
            bk = gk.duplicate()
            gk.scale(-1.0)
            kk.solve(gk, bk)                           # K b = -g
            dpk = (-Nres - tau_u.dot(a)) / (tau_u.dot(bk) + alpha ** 2 * tau_p)
            u.x.petsc_vec.axpy(1.0, a)
            u.x.petsc_vec.axpy(dpk, bk)
            u.x.scatter_forward()
            pval += dpk
        if ok:
            break
        ds_arc *= 0.5
        log(f"[arc] step {k} not converged; ds -> {ds_arc:.3e}")
    if not ok:
        log("[arc] giving up")
        break
    # new tangent: K t_u = -g t_p, oriented along the previous tangent
    Kk = assemble_K()
    kk = lu_solver(Kk)
    gk = assemble_g()
    bk = gk.duplicate()
    gk.scale(-1.0)
    kk.solve(gk, bk)
    tn = np.sqrt(bk.dot(bk) + alpha ** 2)
    new_u = bk.copy()
    new_u.scale(1 / tn)
    new_p = 1.0 / tn
    if new_u.dot(tau_u) + alpha ** 2 * new_p * tau_p < 0:
        new_u.scale(-1.0)
        new_p = -new_p
    tau_u, tau_p = new_u, new_p
    un = u.x.petsc_vec.copy()
    pn = pval
    dF = min_detF()
    arc.append((pn / MMHG, tau_p, dF, it))
    log(f"[arc] {k:3d}: p = {pn / MMHG:9.4f} mmHg, dp/ds sign {np.sign(tau_p):+.0f}, min detF {dF:.4f}, its {it}")
    if it <= 5:
        ds_arc *= 1.3

if comm.rank == 0:
    np.savez(f"sim/out/fem/limit_{name}.npz", hist=np.array(hist), p_star=p_star / MMHG,
             restart=np.array(restart, dtype=object), arc=np.array(arc),
             diag=np.array(diag, dtype=object), alpha=alpha)
log(f"DONE limit {name}: p* = {p_star / MMHG:.4f} mmHg; arc steps {len(arc)}; time {time.time() - t0:.0f}s")
