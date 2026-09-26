"""G_N: the N-leaflet two-circle family, ready for use (front A7).

Construction (McKee, Oliveira, Cuminato et al. 2026, generalised to N leaflets; CLAIMS.md C-001, C-012...C-016,
C-021, C-024, C-027, C-031, C-032).  Ring radius R, height H, theta = 2 pi / N, section parameter z in [0, 1]
(b = z/2, a = 1 - b; the paper's linear profile).  Leaflet 0 consists of two pieces over
Omega = {(t, z) : 0 <= z <= 1, pi <= t <= t_M(z)}:
    piece 1:  X1(t, z) = ( R (-b + a cos t),  R a sin t,  H z )                      (eqs. 38-39, scaled)
    piece 2:  X2(t, z) = Rot_theta ( R (-a + b cos t),  R b sin t,  H z )             (eqs. 40-41 for N = 3)
They meet along the junction M(z) = X1(t_M(z), z) = X2(t_M(z), z) with a crease (fold angle delta(z) > 0 for N >= 3;
none for N = 2).  Leaflet k is Rot_{k theta} of leaflet 0.

Orientation: n1 = X1_t x X1_z / |.| on piece 1 and n2 = -(X2_t x X2_z) / |.| on piece 2 (a consistent orientation of the
leaflet; C-003).  Singular points (excluded): the apex of piece 2 at z = 0 (the base of the commissure post, where
X2_t = 0); the fold angle at z = 0 is the limit z -> 0+.

Criteria for comparing N (Section 6 of the note; the opening area is a response, not imposed):
    "paper"   : R and H given (N = 3, R = 10 mm, H = 15 mm has the dimensions of the paper's models; that their shape is
                this family with h = H/R = 1.5, i.e. a linear vertical coordinate, is an assumption, not confirmed);
    "fixed_R" : R given; H chosen so that the TOTAL leaflet area equals S;
    "fixed_H" : H given; R chosen so that the total leaflet area equals S.
Scaling law (C-031): Area(N, R, H) = R^2 * Atot_N(H / R); fixing R or fixing H selects DIFFERENT shapes (C-031, C-032).
Usage:
    from wheatley.gn import family
    G = family(4, "fixed_R", R=10.0, S=family(3, "paper", R=10.0, H=15.0).area())
    G.X1(t, z); G.normal(1, t, z); G.first_form(2, t, z); G.fold_angle_deg(z); G.surface_mesh(40, 60).write("g4.vtk")
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from . import nleaflet as nl


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def total_area_normalised(N: int, h: float) -> float:
    """Total area of the N leaflets for R = 1 and height h (C-024 with g' = 1):
    N/2 int_0^1 int_pi^{t_M(z)} sqrt(4 h^2 + (1 + cos t)^2) dt dz."""
    inner = lambda z: quad(lambda t: np.sqrt(4 * h * h + (1 + np.cos(t)) ** 2), np.pi, float(nl.t_M(N, z)),
                           epsabs=1e-13, epsrel=1e-13)[0]
    return N / 2 * quad(inner, 0.0, 1.0, epsabs=1e-12, epsrel=1e-12, limit=200)[0]


@dataclass
class GN:
    N: int
    R: float
    H: float
    criterion: str = "paper"
    S: float | None = None                   # target total area (None for "paper")
    theta: float = field(init=False)

    def __post_init__(self):
        if not float(self.N).is_integer() or self.N < 2 or self.R <= 0 or self.H <= 0:
            raise ValueError("need an integer N >= 2 and R, H > 0")
        self.theta = 2 * np.pi / self.N
        self._rot = _rot(self.theta)
        self._grade2 = 1.0      # uniform t on both pieces: the free edges of neighbours coincide exactly at z = 1

    # ------------------------------------------------------------------ geometry
    @property
    def h(self) -> float:
        """Aspect ratio H / R: every dimensionless quantity depends only on (N, h) (C-031)."""
        return self.H / self.R

    def t_M(self, z):
        """Junction parameter (C-012): t_M(0) = pi + theta, t_M(1) = 2 pi."""
        return nl.t_M(self.N, z)

    def X1(self, t, z):
        t, z = np.broadcast_arrays(np.asarray(t, float), np.asarray(z, float))
        b = z / 2
        a = 1 - b
        return np.stack([self.R * (-b + a * np.cos(t)), self.R * a * np.sin(t), self.H * z], axis=-1)

    def X2(self, t, z):
        t, z = np.broadcast_arrays(np.asarray(t, float), np.asarray(z, float))
        b = z / 2
        a = 1 - b
        p = np.stack([self.R * (-a + b * np.cos(t)), self.R * b * np.sin(t), self.H * z], axis=-1)
        return p @ self._rot.T

    def X(self, piece: int, t, z, k: int = 0):
        """Point of piece 1 or 2 of leaflet k."""
        P = self.X1(t, z) if piece == 1 else self.X2(t, z)
        return P @ _rot(k * self.theta).T

    def derivatives(self, piece: int, t, z):
        """Exact (X_t, X_z) of piece 1 or 2 of leaflet 0."""
        t, z = np.broadcast_arrays(np.asarray(t, float), np.asarray(z, float))
        b = z / 2
        a = 1 - b
        R, H = self.R, self.H
        zero = np.zeros_like(t)
        if piece == 1:
            Xt = np.stack([-R * a * np.sin(t), R * a * np.cos(t), zero], axis=-1)
            Xz = np.stack([R * (-0.5 - 0.5 * np.cos(t)), -0.5 * R * np.sin(t), H + zero], axis=-1)
        else:
            Xt = np.stack([-R * b * np.sin(t), R * b * np.cos(t), zero], axis=-1) @ self._rot.T
            Xz = np.stack([R * (0.5 + 0.5 * np.cos(t)), 0.5 * R * np.sin(t), H + zero], axis=-1) @ self._rot.T
        return Xt, Xz

    def first_form(self, piece: int, t, z):
        """(E, F, G) of the first fundamental form in (t, z)."""
        Xt, Xz = self.derivatives(piece, t, z)
        return (np.einsum("...i,...i", Xt, Xt), np.einsum("...i,...i", Xt, Xz), np.einsum("...i,...i", Xz, Xz))

    def normal(self, piece: int, t, z, k: int = 0):
        """Unit normal with the leaflet's consistent orientation (n1 = X_t x X_z, n2 = -(X_t x X_z)).
        Undefined at the apex of piece 2 (z = 0)."""
        Xt, Xz = self.derivatives(piece, t, z)
        n = np.cross(Xt, Xz)
        if piece == 2:
            n = -n
        n = n / np.linalg.norm(n, axis=-1, keepdims=True)
        return n @ _rot(k * self.theta).T

    def junction(self, z):
        """Junction curve M(z) of leaflet 0."""
        return self.X1(self.t_M(z), z)

    def fold_angle_deg(self, z):
        """Fold angle delta(z) along the junction (C-014 / C-021 with h = H/R): angle between n1 and n2 at M(z).
        0 for N = 2 (G1 junction); z = 0 gives the limit z -> 0+ (supremum)."""
        X = 1 + np.cos(self.t_M(z))
        h2c = 4 * self.h ** 2
        c = (X * X - h2c * np.cos(self.theta)) / (h2c + X * X)
        return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))

    def area(self) -> float:
        """Total area of the N leaflets = R^2 * Atot_N(H/R) (C-031)."""
        return self.R ** 2 * total_area_normalised(self.N, self.h)

    def projected_area(self) -> float:
        """Area of the top-view projection of the N leaflets = pi R^2 for every N (C-027)."""
        return np.pi * self.R ** 2

    # ------------------------------------------------------------------ meshing
    def surface_mesh(self, nz: int = 30, nt: int = 40, leaflets=None):
        """Triangulated surface of the valve (all N leaflets by default) as a meshio.Mesh.

        Topology contract:
          - each leaflet is its own surface; leaflets are NOT merged where they touch (the commissure posts and, at z = 1,
            the free edges of neighbours coincide geometrically but have separate vertices); the surface is not watertight;
          - within a leaflet, pieces 1 and 2 share the junction vertices (conforming along the crease) and the apex of
            piece 2 is the vertex M(0) (base of the post);
          - triangles are oriented with the leaflet's consistent normal (n1 on piece 1, n2 = -(X_t x X_z) on piece 2), so
            the two faces at a crease edge traverse it in opposite directions;
          - quad diagonals are chosen so that no triangle has all three vertices on a curve shared with a neighbouring
            leaflet (which produced coincident facets at the top of the posts);
          - contact: along the posts the two leaflets' exact surfaces are tangent, and with the same uniform step in t the
            first strips of triangles are coplanar and nested (a chord from the tangency point with step eps makes the angle
            eps/2 with the tangent whatever the radius).  A graded step
            removed that overlap but produced crossings near the free edges and broke their coincidence, so the
            uniform step is kept and the overlap is documented: all inter-leaflet intersections lie within one element
            of the post lines (tests).
        Data: cell 'leaflet', 'piece', 'normal' (exact unit normal of the piece at the parametric centre of the grid quad containing the triangle);
        point 'normal1', 'normal2' (exact normal of piece 1 / piece 2 at the vertex, NaN where the vertex is not on that
        piece or the normal is undefined: the apex of piece 2), 'on_crease' (1 on the junction M).
        Use VTK (or another format with data): STL keeps only the geometry and loses all of these fields."""
        import meshio
        leaflets = range(self.N) if leaflets is None else leaflets
        zs = np.linspace(0.0, 1.0, nz + 1)
        pts, tris, lf, pc, cn, n1s, n2s, crease = [], [], [], [], [], [], [], []
        nan3 = np.full(3, np.nan)
        for k in leaflets:
            grid = {}
            for piece in (1, 2):
                for iz, z in enumerate(zs):
                    sj = np.linspace(0.0, 1.0, nt + 1)
                    if piece == 2:
                        sj = sj ** self._grade2   # piece-2 grading at the post (see surface_mesh docstring)
                    ts = np.pi + (float(self.t_M(z)) - np.pi) * sj
                    for it, t in enumerate(ts):
                        if piece == 2 and (it == nt or iz == 0):   # junction (and apex = M(0)): shared with piece 1
                            vid = grid[(1, iz, nt)] if it == nt else grid[(1, 0, nt)]
                            grid[(2, iz, it)] = vid
                            if it == nt and iz > 0:
                                n2s[vid] = self.normal(2, t, z, k)
                            continue
                        grid[(piece, iz, it)] = len(pts)
                        pts.append(self.X(piece, t, z, k))
                        n1s.append(self.normal(1, t, z, k) if piece == 1 else nan3)
                        n2s.append(self.normal(2, t, z, k) if piece == 2 else nan3)
                        crease.append(1 if it == nt else 0)
                for iz in range(nz):
                    for it in range(nt):
                        a, b_, c, d_ = (grid[(piece, iz, it)], grid[(piece, iz, it + 1)],
                                        grid[(piece, iz + 1, it + 1)], grid[(piece, iz + 1, it)])
                        # diagonal b-d: no triangle (post vertex, two top-row vertices) shared with a neighbour
                        for tri in ((a, b_, d_), (b_, c, d_)):
                            if len(set(tri)) < 3:
                                continue
                            if piece == 2:                  # consistent orientation: n2 = -(X_t x X_z)
                                tri = (tri[0], tri[2], tri[1])
                            tris.append(tri)
                            lf.append(k)
                            pc.append(piece)
                            zc = (iz + 0.5) / nz
                            sc = (it + 0.5) / nt
                            if piece == 2:
                                sc = sc ** self._grade2
                            tcen = np.pi + sc * (float(self.t_M(zc)) - np.pi)
                            cn.append(self.normal(piece, tcen, zc, k))
        pts = np.array(pts)
        return meshio.Mesh(pts, [("triangle", np.array(tris))],
                           point_data={"normal1": np.array(n1s), "normal2": np.array(n2s),
                                       "on_crease": np.array(crease)},
                           cell_data={"leaflet": [np.array(lf)], "piece": [np.array(pc)], "normal": [np.array(cn)]})

    def summary(self) -> dict:
        zs = np.concatenate([[0.0], np.linspace(1e-3, 1.0, 400)])
        f = self.fold_angle_deg(zs)
        return dict(N=self.N, criterion=self.criterion, R=self.R, H=self.H, h=self.h, area=self.area(),
                    fold_min_deg=float(f[1:].min()), fold_sup_deg=float(f[0]))


def family(N: int, criterion: str = "paper", R: float | None = None, H: float | None = None,
           S: float | None = None) -> GN:
    """Build G_N for a comparison criterion (see module docstring).
    paper:   R and H required.   fixed_R: R and S required.   fixed_H: H and S required."""
    if criterion == "paper":
        if R is None or H is None:
            raise ValueError("paper: give R and H")
        return GN(N, R, H, "paper")
    if S is None or S <= 0:
        raise ValueError("give the target total area S")
    if criterion == "fixed_R":
        if R is None:
            raise ValueError("fixed_R: give R")
        s = S / R ** 2
        if s <= np.pi:
            raise ValueError("fixed_R: S must exceed pi R^2 (C-027: the area tends to pi R^2 as H -> 0)")
        f = lambda hh: total_area_normalised(N, hh) - s
        hi = 1.0
        while f(hi) < 0:                        # A_N is increasing in h and unbounded: expand until the root is bracketed
            hi *= 2.0
        h = brentq(f, 0.0, hi, xtol=1e-14 * max(1.0, hi))   # f(0) = pi - s < 0 (C-027)
        return GN(N, R, h * R, "fixed_R", S)
    if criterion == "fixed_H":
        if H is None:
            raise ValueError("fixed_H: give H")
        f = lambda r: r * r * total_area_normalised(N, H / r) - S     # increasing in r (C-032), 0 at r -> 0
        lo, hi = H, H
        while f(hi) < 0:
            hi *= 2.0
        while f(lo) > 0:
            lo /= 2.0
        Rn = brentq(f, lo, hi, xtol=1e-15 * hi)
        return GN(N, Rn, H, "fixed_H", S)
    raise ValueError(f"unknown criterion {criterion!r}")


def table(Ns=(2, 3, 4, 5, 6, 7), criterion="fixed_R", ref=(3, 10.0, 15.0), R=None, H=None):
    """Parameters and fold angles for each N at the area of the reference design ref = (N, R, H)."""
    S = family(ref[0], "paper", R=ref[1], H=ref[2]).area()
    rows = []
    for N in Ns:
        G = family(N, criterion, R=R if R is not None else ref[1], H=H if H is not None else ref[2], S=S)
        rows.append(G.summary())
    return rows


def intersecting_triangle_pairs(P, T, group, eps_rel=1e-9):
    """Pairs of triangles from different groups (e.g. leaflets) whose interiors intersect: separating-axis test with the two
    normals, the 9 edge cross products and, for coplanar pairs, only the 6 in-plane edge normals (coplanar pairs give zero-
    thickness projections on the normals, which must not count as separation).  Touching (shared edge/vertex) is not an
    intersection.  Returns an array of index pairs."""
    A = P[T]; lo, hi = A.min(1), A.max(1); h = np.median(np.linalg.norm(A[:,1]-A[:,0],axis=1)); eps = eps_rel*h
    cell = 2*h; ijk = np.floor(lo/cell).astype(int); ijk2 = np.floor(hi/cell).astype(int)
    buckets = {}
    for k in range(len(T)):
        for a in range(ijk[k,0], ijk2[k,0]+1):
            for b in range(ijk[k,1], ijk2[k,1]+1):
                for c in range(ijk[k,2], ijk2[k,2]+1):
                    buckets.setdefault((a,b,c),[]).append(k)
    pairs=set()
    for ks in buckets.values():
        ks=np.array(ks)
        if len(ks)<2: continue
        i,j=np.triu_indices(len(ks),1); a,b=ks[i],ks[j]
        ok=(group[a]!=group[b])&np.all((lo[a]<=hi[b]+eps)&(lo[b]<=hi[a]+eps),axis=1)
        pairs.update(zip(a[ok].tolist(),b[ok].tolist()))
    pairs=np.array(sorted(pairs)); 
    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=int)
    X,Y=A[pairs[:,0]],A[pairs[:,1]]
    eX=np.stack([X[:,1]-X[:,0],X[:,2]-X[:,1],X[:,0]-X[:,2]],1); eY=np.stack([Y[:,1]-Y[:,0],Y[:,2]-Y[:,1],Y[:,0]-Y[:,2]],1)
    nX=np.cross(eX[:,0],eX[:,1]); nY=np.cross(eY[:,0],eY[:,1])
    axes=[nX[:,None],nY[:,None],np.cross(eX[:,:,None],eY[:,None,:]).reshape(-1,9,3),np.cross(nX[:,None],eX),np.cross(nY[:,None],eY)]
    ax=np.concatenate(axes,1); nr=np.linalg.norm(ax,axis=2,keepdims=True); valid=nr[...,0]>1e-12*h*h; ax=ax/np.where(nr>0,nr,1)
    px=np.einsum('nkd,nvd->nkv',ax,X); py=np.einsum('nkd,nvd->nkv',ax,Y)
    ov=np.minimum(px.max(2),py.max(2))-np.maximum(px.min(2),py.min(2))
    # coplanar pairs: the normal axes give zero-thickness projections (overlap 0) and must not count as separating
    dY=np.abs(np.einsum('nd,nvd->nv',nX/np.linalg.norm(nX,axis=1,keepdims=True),Y-X[:,[0]])).max(1)
    dX=np.abs(np.einsum('nd,nvd->nv',nY/np.linalg.norm(nY,axis=1,keepdims=True),X-Y[:,[0]])).max(1)
    cop=(dY<=eps)|(dX<=eps)
    valid[cop,:11]=False     # coplanar: only the in-plane edge normals (axes 11..16) are meaningful
    # separated along an axis iff the projected intervals are disjoint or only touch at an end point.  (The earlier test
    # 'overlap <= eps' counted a zero-thickness projection lying strictly inside the other interval as separation, which
    # misses transversal crossings.)
    touch=(px.max(2)<=py.min(2)+eps)|(py.max(2)<=px.min(2)+eps)
    sep=np.any(valid&touch,axis=1)
    return pairs[~sep]
