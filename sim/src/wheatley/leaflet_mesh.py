"""Leaflet sections, G1 fillet of the crease, and thin-solid meshes (front A2 constructive / A5).

Section of leaflet 0 at height z (normalised, ring radius 1): arc of C1 (centre (-b,0), radius 1-b,
t in [pi, t_M]) followed by arc of C2 (centre R_theta(-(1-b),0), radius b, t from t_M down to pi).
They meet at M with an in-plane kink (the fold angle of C-014 seen in a horizontal section).

G1 rounding (C-019): in every horizontal section, replace a neighbourhood of M by a circular fillet of
radius rho(z) tangent to C1 and C2.  Each section is then a G1 curve and the section family varies
smoothly with z, so the surface is G1 (the tangent plane at a tangency point contains the common
section tangent and the tangent of the tangency curve).  For the sampled radii, numerical curvature
evaluation (`fillet_strip_gaussian_curvature`) shows that this particular fillet is not developable.

Physical scaling: x, y multiplied by `radius`; z multiplied by `radius * h` (A-13: the paper's
models are 20 mm x 15 mm, i.e. h = 1.5, unconfirmed).
"""
from __future__ import annotations

import numpy as np

from . import geometry as g


def _circles(z, N=3):
    th = 2 * np.pi / N
    b = z / 2.0
    c1, R1 = np.array([-b, 0.0]), 1.0 - b
    c, s = np.cos(th), np.sin(th)
    Rm = np.array([[c, -s], [s, c]])
    c2, R2 = Rm @ np.array([-(1.0 - b), 0.0]), b
    return c1, R1, c2, R2, Rm


def t_M(z, N=3):
    from . import nleaflet as nl
    return float(nl.t_M(N, z))


def _arc_param(p, c, R, Rm=None):
    v = (p - c) / R
    if Rm is not None:
        v = Rm.T @ v
    a = np.arctan2(v[1], v[0])
    return a % (2 * np.pi) if a % (2 * np.pi) >= np.pi - 1e-12 else a % (2 * np.pi) + 2 * np.pi


def fillet(z, rho, N=3):
    """Return (t1, t2, F, T1, T2, phi0, phi1): tangency parameters on C1/C2, fillet centre, tangency
    points and the fillet arc (angles phi0 -> phi1 about F).  rho > 0, z in (0, 1]."""
    c1, R1, c2, R2, Rm = _circles(z, N)
    tm = t_M(z, N)
    M = c1 + R1 * np.array([np.cos(tm), np.sin(tm)])
    best = None
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            r1, r2 = R1 + s1 * rho, R2 + s2 * rho
            d = np.linalg.norm(c2 - c1)
            if r1 <= 0 or r2 <= 0 or d > r1 + r2 or d < abs(r1 - r2):
                continue
            a = (r1**2 - r2**2 + d**2) / (2 * d)
            hh = np.sqrt(max(r1**2 - a**2, 0.0))
            e = (c2 - c1) / d
            base = c1 + a * e
            for sg in (-1, 1):
                F = base + sg * hh * np.array([-e[1], e[0]])
                T1 = c1 + R1 * (F - c1) / np.linalg.norm(F - c1)
                T2 = c2 + R2 * (F - c2) / np.linalg.norm(F - c2)
                t1 = _arc_param(T1, c1, R1)
                t2 = _arc_param(T2, c2, R2, Rm)
                if not (np.pi <= t1 <= tm + 1e-12 and np.pi <= t2 <= tm + 1e-12):
                    continue
                cost = (tm - t1) + (tm - t2)
                if best is None or cost < best[0]:
                    best = (cost, t1, t2, F, T1, T2)
    if best is None:
        raise ValueError(f"no fillet found at z={z}, rho={rho}")
    _, t1, t2, F, T1, T2 = best
    p0 = np.arctan2(*(T1 - F)[::-1])
    p1 = np.arctan2(*(T2 - F)[::-1])
    # choose the arc (p0 -> p1, either way round) whose midpoint is closer to M
    cand = []
    for dphi in ((p1 - p0) % (2 * np.pi), (p1 - p0) % (2 * np.pi) - 2 * np.pi):
        mid = F + rho * np.array([np.cos(p0 + dphi / 2), np.sin(p0 + dphi / 2)])
        cand.append((np.linalg.norm(mid - M), dphi))
    dphi = min(cand)[1]
    return t1, t2, F, T1, T2, p0, p0 + dphi


def section(z, n1, n2, nf=0, rho=0.0, N=3):
    """Points (n1 + nf + n2 + 2, 2) of the section at height z: arc 1 (n1 segments), nf fillet points,
    arc 2 (n2 segments).  rho = 0 (or z = 0): sharp crease; the nf fillet slots and the first point of
    arc 2 all sit at M (coincident points are merged by `triangulate`)."""
    c1, R1, c2, R2, Rm = _circles(z, N)
    tm = t_M(z, N)
    if rho <= 0 or z <= 0:
        t1 = t2 = tm
        Fp = np.zeros((0, 2))
    else:
        t1, t2, F, T1, T2, p0, p1 = fillet(z, rho, N)
        ph = np.linspace(p0, p1, nf + 2)[1:-1]
        Fp = F + rho * np.stack([np.cos(ph), np.sin(ph)], -1)
    ta = np.linspace(np.pi, t1, n1 + 1)
    A1 = c1 + R1 * np.stack([np.cos(ta), np.sin(ta)], -1)
    if len(Fp) == 0:
        Fp = np.repeat(A1[-1:], nf, axis=0)
    tt = np.linspace(t2, np.pi, n2 + 1)
    A2 = c2 + R2 * (np.stack([np.cos(tt), np.sin(tt)], -1) @ Rm.T)
    return np.vstack([A1, Fp, A2])


def _resample(P, m, slots):
    """Resample polyline P by arclength into m+1 distinct points, padded to `slots` by repeating the
    last point (coincident points are merged later, turning quads into fan triangles)."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    if s[-1] <= 1e-14:
        return np.repeat(P[:1], slots, axis=0)
    u = np.linspace(0, s[-1], m + 1)
    Q = np.column_stack([np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1])])
    return np.vstack([Q, np.repeat(Q[-1:], slots - len(Q), axis=0)])


def surface_grid(nz, n1, n2, nf=0, rho0=0.0, N=3, radius=10.0, h=1.5, target=None, rho_min=0.0):
    """Grid (nz+1, n1+nf+n2+2, 3) in physical units.  Fillet radius rho(z) = rho0 * z (normalised).
    The number of distinct points on arc 2 and on the fillet in each row is proportional to their
    length (target spacing = arc-1 spacing at the top row unless `target` is given), which avoids
    slivers near the apex of surface 2.  With rho0 > 0 the fillet radius is sqrt((rho0 z)^2 + rho_min^2) (normalised): a
    fillet thinner than the shell thickness cannot be offset without folding, so callers pass
    rho_min = thickness / radius."""
    fine = 400
    rows = []
    for k in range(nz + 1):
        z = k / nz
        # smooth radius law (max(rho0 z, rho_min) has a kink in z and breaks G1)
        rho_z = float(np.hypot(rho0 * z, rho_min)) if (rho0 > 0 and z > 0) else 0.0
        try:
            P = section(z, fine, fine, fine if rho_z > 0 else 0, rho_z, N)
        except ValueError:                      # the fillet does not fit this low section: keep the sharp crease here
            rho_z = 0.0
            P = section(z, fine, fine, 0, 0.0, N)
        A1, rest = P[:fine + 1], P[fine + 1:]
        nfl = fine if rho_z > 0 else 0
        Fp, A2 = rest[:nfl], rest[nfl:]
        if target is None and k == 0:
            pass
        L1 = np.sum(np.linalg.norm(np.diff(A1, axis=0), axis=1))
        tgt = target if target is not None else L1 / n1
        r1 = _resample(A1, n1, n1 + 1)
        if nf:
            Fseg = np.vstack([A1[-1:], Fp, A2[:1]]) if nfl else np.repeat(A1[-1:], 2, axis=0)
            Lf = np.sum(np.linalg.norm(np.diff(Fseg, axis=0), axis=1))
            if nfl and Lf < tgt / 3:
                # fillet shorter than half an element: use the sharp section in this row
                P = section(z, fine, fine, 0, 0.0, N)
                A1, A2 = P[:fine + 1], P[fine + 1:]
                L1 = np.sum(np.linalg.norm(np.diff(A1, axis=0), axis=1))
                r1 = _resample(A1, n1, n1 + 1)
                Fseg = np.repeat(A1[-1:], 2, axis=0)
                Lf = 0.0
            mf = int(np.clip(round(3 * Lf / tgt), 0, nf + 1))
            rf = _resample(Fseg, max(mf, 1), nf + 2)[1:-1] if mf >= 1 else np.repeat(A1[-1:], nf, axis=0)
        else:
            rf = np.zeros((0, 2))
        L2 = np.sum(np.linalg.norm(np.diff(A2, axis=0), axis=1))
        m2 = int(np.clip(round(L2 / tgt), 1, n2))
        r2 = _resample(A2, m2, n2 + 1)
        Q = np.vstack([r1, rf, r2])
        rows.append(np.column_stack([radius * Q, np.full(len(Q), radius * h * z)]))
    return np.array(rows)


# ---------------------------------------------------------------- thin-solid mesh
_ROT = {0: (0, 1, 2, 3, 4, 5), 1: (1, 2, 0, 4, 5, 3), 2: (2, 0, 1, 5, 3, 4),
        3: (3, 5, 4, 0, 2, 1), 4: (4, 3, 5, 1, 0, 2), 5: (5, 4, 3, 2, 1, 0)}


def _split_prism(v):
    """Dompierre et al. (1999) conforming split of a prism (v0,v1,v2 bottom; v3,v4,v5 top) into 3 tets."""
    k = int(np.argmin(v))
    r = [v[i] for i in _ROT[k]]
    if min(r[1], r[5]) < min(r[2], r[4]):
        return [(r[0], r[1], r[2], r[5]), (r[0], r[1], r[5], r[4]), (r[0], r[4], r[5], r[3])]
    return [(r[0], r[1], r[2], r[4]), (r[0], r[4], r[2], r[5]), (r[0], r[4], r[5], r[3])]


def triangulate(G, tol=1e-9, skip=None):
    """Merge coincident grid points and return (X (nv,3), tris (nt,3), idx (grid -> vertex)).
    skip: optional bool array (nr-1, nc-1); quads with skip[k, i] are not triangulated (used to truncate the apex of
    surface 2, see `apex_skip`). Vertices left unused are removed."""
    nr, nc, _ = G.shape
    P = G.reshape(-1, 3)
    key = np.round(P / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    X = P[first[order]]
    idx = remap[inv.ravel()].reshape(nr, nc)
    tris = []
    for k in range(nr - 1):
        for i in range(nc - 1):
            if skip is not None and skip[k, i]:
                continue
            a, b_, c, d = idx[k, i], idx[k, i + 1], idx[k + 1, i + 1], idx[k + 1, i]
            for tri in ((a, b_, c), (a, c, d)):
                if len(set(tri)) == 3:
                    p0, p1, p2 = X[list(tri)]
                    if np.linalg.norm(np.cross(p1 - p0, p2 - p0)) > 1e-14:
                        tris.append(tri)
    tris = np.array(tris)
    used = np.unique(tris)
    if len(used) < len(X):
        remap = -np.ones(len(X), dtype=np.int64)
        remap[used] = np.arange(len(used))
        X, tris, idx = X[used], remap[tris], remap[idx]      # idx = -1 for grid points no longer in the mesh
    return X, tris, idx


def corner_skip(G, n1, zc=0.125):
    """Quads to skip so that the corner at the base of post P is removed identically for the sharp and the rounded
    leaflet: in the rows with normalised height z < zc, the junction/fillet slots and arc 2 are not meshed.  Near P the
    section radius of cone 2 (z/2 in normalised units) is comparable to the shell thickness and the rounded section
    turns by almost 180 degrees within half a millimetre, so no thin solid of the given thickness fits
    (twisted prisms and overlapping tets).  zc * (number of rows) must be an integer, so
    that the cut is the same physical region on every mesh.  Returns (skip, kcut)."""
    nr, nc, _ = G.shape
    nz = nr - 1
    kcut = zc * nz
    if abs(kcut - round(kcut)) > 1e-9:
        raise ValueError(f"corner_skip: zc * nz = {kcut} is not an integer; choose nz as a multiple of {1 / zc:g}")
    kcut = int(round(kcut))
    skip = np.zeros((nr - 1, nc - 1), dtype=bool)
    skip[:kcut, n1:] = True
    return skip, kcut


def vertex_normals(X, tris):
    n = np.zeros_like(X)
    fn = np.cross(X[tris[:, 1]] - X[tris[:, 0]], X[tris[:, 2]] - X[tris[:, 0]])
    for j in range(3):
        np.add.at(n, tris[:, j], fn)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def thin_solid(X, tris, thickness=0.3, layers=2, global_check=True):
    """Offset the triangulated mid-surface by +-thickness/2 along vertex normals (orientation: the
    grid's du x dz, i.e. +n1 on surface 1 and -n2 on surface 2 = towards the aorta/outside) and split
    the prisms into tetrahedra.  Returns (Y (nv*(L+1),3), tets, n, layer_of_node)."""
    n = vertex_normals(X, tris)
    nv = len(X)
    zs = np.linspace(-thickness / 2, thickness / 2, layers + 1)
    Y = np.vstack([X + z * n for z in zs])
    tets = []
    for l in range(layers):
        for (a, b_, c) in tris:
            v = [a + l * nv, b_ + l * nv, c + l * nv, a + (l + 1) * nv, b_ + (l + 1) * nv, c + (l + 1) * nv]
            tets.extend(_split_prism(v))
    tets = np.array(tets, dtype=np.int64)
    p = Y[tets]
    vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    # Every prism must give three tets of the same sign (else it is twisted/folded and the split is invalid).
    # A prism of consistently negative sign only has the opposite orientation, which a vertex swap fixes.
    sg = np.sign(vol).reshape(-1, 3)
    if np.any(sg.min(axis=1) != sg.max(axis=1)) or np.any(sg == 0):
        bad = np.where((sg.min(axis=1) != sg.max(axis=1)) | np.any(sg == 0, axis=1))[0]
        raise ValueError(f"thin_solid: {len(bad)} twisted or degenerate prisms (first prism index {bad[0]})")
    tets[vol < 0] = tets[vol < 0][:, [0, 2, 1, 3]]
    n_bad = count_same_side_faces(Y, tets)
    if n_bad:
        raise ValueError(f"thin_solid: {n_bad} interior faces with both neighbouring tets on the same side (overlap)")
    if global_check:
        n_ov, pair = count_overlapping_tets(Y, tets)
        if n_ov:
            raise ValueError(f"thin_solid: {n_ov} pairs of tetrahedra overlap in their interiors (first {pair})")
    layer = np.repeat(np.arange(layers + 1), nv)
    return Y, tets, n, layer


def count_same_side_faces(Y, tets):
    """Number of interior faces whose two tetrahedra lie on the same side (overlapping elements)."""
    import itertools
    faces, opp = [], []
    for ix in itertools.combinations(range(4), 3):
        faces.append(np.sort(tets[:, ix], axis=1))
        opp.append(tets[:, list(set(range(4)) - set(ix))[0]])
    faces, opp = np.concatenate(faces), np.concatenate(opp)
    _, inv, cnt = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
    p = Y[faces]
    d = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), Y[opp] - p[:, 0])
    sgn = np.bincount(inv.ravel(), weights=np.sign(d))
    return int(np.sum((cnt == 2) & (np.abs(sgn) == 2)))


def count_overlapping_tets(Y, tets, rel_eps=1e-9, chunk=200000):
    """Exact global check for interior overlap between ANY two tetrahedra (`count_same_side_faces` only
    sees face neighbours).  Candidate pairs: axis-aligned bounding boxes that overlap (uniform grid hashing).  Test:
    separating axis theorem with the 4 + 4 face normals and the 6 x 6 edge cross products; the pair overlaps in its
    interior iff every axis gives an overlap larger than rel_eps * (mesh size), so tets that only touch (shared face, edge
    or vertex) are not counted.  Returns (number of overlapping pairs, first pair or None)."""
    P = Y[tets]                                            # (m, 4, 3)
    lo, hi = P.min(axis=1), P.max(axis=1)
    h = float(np.median(np.max(hi - lo, axis=1)))
    eps = rel_eps * h
    cell = 2.0 * h
    ijk_lo = np.floor(lo / cell).astype(np.int64)
    ijk_hi = np.floor(hi / cell).astype(np.int64)
    buckets = {}
    for k in range(len(tets)):
        for a in range(ijk_lo[k, 0], ijk_hi[k, 0] + 1):
            for b in range(ijk_lo[k, 1], ijk_hi[k, 1] + 1):
                for c in range(ijk_lo[k, 2], ijk_hi[k, 2] + 1):
                    buckets.setdefault((a, b, c), []).append(k)
    pairs = set()
    for ks in buckets.values():
        ks = np.array(ks)
        if len(ks) < 2:
            continue
        ii, jj = np.triu_indices(len(ks), 1)
        A, B = ks[ii], ks[jj]
        ok = np.all((lo[A] <= hi[B] + eps) & (lo[B] <= hi[A] + eps), axis=1)
        pairs.update(zip(np.minimum(A[ok], B[ok]).tolist(), np.maximum(A[ok], B[ok]).tolist()))
    if not pairs:
        return 0, None
    pr = np.array(sorted(pairs))
    fidx = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    eidx = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    count, first = 0, None
    for s0 in range(0, len(pr), chunk):
        a, b = P[pr[s0:s0 + chunk, 0]], P[pr[s0:s0 + chunk, 1]]
        axes = []
        for T in (a, b):
            f = T[:, fidx]
            axes.append(np.cross(f[:, :, 1] - f[:, :, 0], f[:, :, 2] - f[:, :, 0]))
        ea = a[:, eidx[:, 1]] - a[:, eidx[:, 0]]
        eb = b[:, eidx[:, 1]] - b[:, eidx[:, 0]]
        axes.append(np.cross(ea[:, :, None, :], eb[:, None, :, :]).reshape(len(a), 36, 3))
        ax = np.concatenate(axes, axis=1)                   # (n, 44, 3)
        nrm = np.linalg.norm(ax, axis=2, keepdims=True)
        valid = nrm[..., 0] > 1e-12 * h * h
        ax = ax / np.where(nrm > 0, nrm, 1.0)
        pa = np.einsum("nkd,nvd->nkv", ax, a)
        pb = np.einsum("nkd,nvd->nkv", ax, b)
        overlap = np.minimum(pa.max(2), pb.max(2)) - np.maximum(pa.min(2), pb.min(2))
        separated = np.any(valid & (overlap <= eps), axis=1)
        bad = np.where(~separated)[0]
        count += len(bad)
        if first is None and len(bad):
            first = tuple(pr[s0 + bad[0]].tolist())
    return count, first


def fillet_strip_gaussian_curvature(rho0, nz=60, N=3, h=1.0):
    """Max |K| on the fillet strip of the normalised leaflet (finite differences on the fillet
    parametrisation (s, z), s in [0,1] along the fillet arc)."""
    def P(s, z):
        t1, t2, F, T1, T2, p0, p1 = fillet(z, rho0 * z, N)
        ph = p0 + s * (p1 - p0)
        q = F + rho0 * z * np.array([np.cos(ph), np.sin(ph)])
        return np.array([q[0], q[1], h * z])
    out = []
    e = 1e-4
    for z in np.linspace(0.1, 0.95, nz):
        for s in (0.25, 0.5, 0.75):
            Xs = (P(s + e, z) - P(s - e, z)) / (2 * e)
            Xz = (P(s, z + e) - P(s, z - e)) / (2 * e)
            Xss = (P(s + e, z) - 2 * P(s, z) + P(s - e, z)) / e**2
            Xzz = (P(s, z + e) - 2 * P(s, z) + P(s, z - e)) / e**2
            Xsz = (P(s + e, z + e) - P(s + e, z - e) - P(s - e, z + e) + P(s - e, z - e)) / (4 * e * e)
            nn = np.cross(Xs, Xz)
            nn /= np.linalg.norm(nn)
            E, F_, G_ = Xs @ Xs, Xs @ Xz, Xz @ Xz
            L, M_, Nn = Xss @ nn, Xsz @ nn, Xzz @ nn
            out.append((z, s, (L * Nn - M_ * M_) / (E * G_ - F_ * F_)))
    return np.array(out)
