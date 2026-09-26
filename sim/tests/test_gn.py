"""A7: the G_N package (sim/src/wheatley/gn.py) against the accepted claims."""
import numpy as np
import pytest

from wheatley import gn
from wheatley import nleaflet as nl

REF = gn.family(3, "paper", R=10.0, H=15.0)


def fd(f, t, z, h=1e-6):
    return (f(t + h, z) - f(t - h, z)) / (2 * h), (f(t, z + h) - f(t, z - h)) / (2 * h)


@pytest.mark.parametrize("N", [2, 3, 4, 6])
@pytest.mark.parametrize("R,H", [(1.0, 1.0), (10.0, 15.0), (7.0, 4.0)])
def test_derivatives_normals_first_form(N, R, H):
    G = gn.GN(N, R, H)
    rng = np.random.default_rng(N)
    for t, z in zip(rng.uniform(np.pi, 2 * np.pi, 15), rng.uniform(0.05, 1.0, 15)):
        for piece, X in ((1, G.X1), (2, G.X2)):
            Xt, Xz = G.derivatives(piece, t, z)
            ft, fz = fd(X, t, z)
            assert np.allclose(Xt, ft, atol=1e-6 * R) and np.allclose(Xz, fz, atol=1e-6 * max(R, H))
            E, F, Gg = G.first_form(piece, t, z)
            assert np.isclose(E, ft @ ft, rtol=1e-6) and np.isclose(Gg, fz @ fz, rtol=1e-6)
            n = G.normal(piece, t, z)
            assert np.isclose(np.linalg.norm(n), 1) and abs(n @ ft) < 1e-6 * R and abs(n @ fz) < 1e-6 * max(R, H)


@pytest.mark.parametrize("N", [2, 3, 4, 5])
def test_junction_and_fold_angle(N):
    G = gn.GN(N, 10.0, 12.0)
    for z in np.linspace(0.05, 1.0, 9):
        tm = float(G.t_M(z))
        assert np.allclose(G.X1(tm, z), G.X2(tm, z), atol=1e-9)
        ang = np.degrees(np.arccos(np.clip(G.normal(1, tm, z) @ G.normal(2, tm, z), -1, 1)))
        assert np.isclose(ang, G.fold_angle_deg(z), atol=1e-4)     # arccos of a dot product near 1: ~sqrt(eps)
    if N == 2:
        assert np.allclose(G.fold_angle_deg(np.linspace(0, 1, 11)), 0.0, atol=1e-6)


def test_fold_angle_matches_accepted_claims():
    G1 = gn.GN(3, 1.0, 1.0)                                     # C-003
    assert np.isclose(G1.fold_angle_deg(1.0), np.degrees(np.arccos(0.75)))
    assert np.isclose(G1.fold_angle_deg(0.0), np.degrees(np.arccos(17 / 25)))
    G15 = gn.GN(3, 10.0, 15.0)                                  # C-021, h = 3/2
    assert np.isclose(G15.fold_angle_deg(1.0), np.degrees(np.arccos(17 / 26)))
    assert np.isclose(G15.fold_angle_deg(0.0), np.degrees(np.arccos(3 / 5)))


def test_free_edges_close():                                    # C-013
    G = gn.GN(4, 10.0, 15.0)
    t = np.linspace(np.pi, 2 * np.pi, 50)
    assert np.allclose(G.X(2, t, 1.0, k=0), G.X(1, t, 1.0, k=1), atol=1e-9)


def test_area_scaling_and_reference():                          # C-031, C-015
    assert np.isclose(gn.total_area_normalised(3, 1.0), 3 * nl.leaflet_area(3), rtol=1e-8)
    G = gn.GN(4, 7.0, 5.0)
    assert np.isclose(G.area(), 49.0 * gn.total_area_normalised(4, 5.0 / 7.0), rtol=1e-12)


@pytest.mark.parametrize("N", [2, 4, 5])
def test_equal_area_families(N):                                # C-032 numbers (h_ref = 1)
    S = gn.family(3, "paper", R=1.0, H=1.0).area()
    GR = gn.family(N, "fixed_R", R=1.0, S=S)
    GH = gn.family(N, "fixed_H", H=1.0, S=S)
    assert np.isclose(GR.area(), S, rtol=1e-9) and np.isclose(GH.area(), S, rtol=1e-9)
    expected = {2: (1.24521, 0.87309), 4: (0.85419, 1.12102), 5: (0.75833, 1.23080)}[N]
    assert np.isclose(GR.h, expected[0], atol=1e-5) and np.isclose(GH.h, expected[1], atol=1e-5)


def test_fixed_R_rejects_area_below_projection():               # C-027
    with pytest.raises(ValueError):
        gn.family(3, "fixed_R", R=1.0, S=np.pi * 0.99)


@pytest.mark.parametrize("N,nz,nt", [(3, 12, 16), (2, 20, 30), (4, 40, 60), (7, 20, 30)])
def test_surface_mesh_contract(N, nz, nt):
    G = gn.GN(N, 10.0, 15.0)
    m = G.surface_mesh(nz, nt)
    P, T = m.points, m.cells_dict["triangle"]
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    area2 = np.linalg.norm(fn, axis=1)
    assert area2.min() > 0                                        # no degenerate triangles
    fn /= area2[:, None]
    # 1. orientation agrees with the exact normal of the piece (cell data)
    assert np.all(np.einsum("ij,ij->i", fn, m.cell_data["normal"][0]) > 0.9)
    # 2. within a leaflet every edge is used at most twice, and interior edges in opposite directions (incl. the crease)
    lf = m.cell_data["leaflet"][0]
    for k in range(N):
        Tk = T[lf == k]
        e = np.vstack([Tk[:, [0, 1]], Tk[:, [1, 2]], Tk[:, [2, 0]]])
        key, cnt = np.unique(np.sort(e, axis=1), axis=0, return_counts=True)
        assert cnt.max() <= 2
        directed = set(map(tuple, e.tolist()))
        assert len(directed) == len(e)                            # no directed edge twice => consistent orientation
    # 3. no two triangles (anywhere) with the same three vertex positions
    q = np.round(P / 1e-9).astype(np.int64)
    keys = np.sort(np.array([[hash(tuple(q[v])) for v in tri] for tri in T]), axis=1)
    _, cnt = np.unique(keys, axis=0, return_counts=True)
    assert cnt.max() == 1
    # 3b. intersections between leaflets only in the documented band around the posts
    bad = gn.intersecting_triangle_pairs(P, T, lf)
    if len(bad):
        # every point of an overlapping triangle within one element of ONE post line: for each triangle, the distance to a
        # vertical line is convex, so its max over the triangle is at a vertex; take the best single post per triangle
        # (a per-vertex nearest post is not enough) and the largest edge over all three edges
        posts = np.array([[G.R * np.cos(np.pi + k * G.theta), G.R * np.sin(np.pi + k * G.theta)] for k in range(N)])
        tri = P[T[np.unique(bad)]][:, :, :2]                                      # (m, 3, 2)
        dmax = np.linalg.norm(tri[:, :, None, :] - posts[None, None], axis=3).max(axis=1)   # (m, posts): max over vertices
        worst = dmax.min(axis=1).max()
        hmax = max(np.linalg.norm(P[T[:, i]] - P[T[:, j]], axis=1).max() for i, j in ((0, 1), (1, 2), (0, 2)))
        assert worst <= 1.0001 * hmax
    # 3c. the free edges of neighbours coincide exactly (same vertex positions at z = 1)
    top = P[np.isclose(P[:, 2], G.H)]
    q = np.round(top / 1e-9).astype(np.int64)
    _, cnt = np.unique(q, axis=0, return_counts=True)
    assert np.all(cnt >= 2)
    # 4. normals per piece: both on the crease (except the apex), unit length where defined
    n1, n2, cr = m.point_data["normal1"], m.point_data["normal2"], m.point_data["on_crease"]
    ok1, ok2 = ~np.isnan(n1[:, 0]), ~np.isnan(n2[:, 0])
    assert np.allclose(np.linalg.norm(n1[ok1], axis=1), 1) and np.allclose(np.linalg.norm(n2[ok2], axis=1), 1)
    assert np.sum(cr == 1) == N * (nz + 1)
    assert np.sum((cr == 1) & ok1 & ok2) == N * nz                 # all crease vertices but the apex carry both normals
    if N >= 3:
        both = (cr == 1) & ok1 & ok2
        ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", n1[both], n2[both]), -1, 1)))
        assert ang.min() > 30                                     # the crease angle is visible in the data


def test_surface_mesh_is_conforming():
    G = gn.GN(3, 10.0, 15.0)
    m = G.surface_mesh(12, 16)
    tris = m.cells_dict["triangle"]
    edges = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [0, 2]]]), axis=1)
    _, cnt = np.unique(edges, axis=0, return_counts=True)
    assert cnt.max() <= 2                                         # manifold (each edge in at most two triangles)
    area = 0.5 * np.linalg.norm(np.cross(m.points[tris[:, 1]] - m.points[tris[:, 0]],
                                         m.points[tris[:, 2]] - m.points[tris[:, 0]]), axis=1).sum()
    assert abs(area - G.area()) / G.area() < 0.02                  # coarse mesh area close to the exact area
    assert m.point_data["on_crease"].sum() == 3 * 13               # one junction vertex per row and leaflet


def test_solvers_extreme_targets():                             # fixed brackets failed at extreme targets
    G = gn.family(3, "fixed_R", R=1.0, S=1000.0)
    assert np.isclose(G.H, 128.5663228629, rtol=1e-8)
    G = gn.family(3, "fixed_H", H=1.0, S=1e-7)
    assert np.isclose(G.R, 1.2856733858e-8, rtol=1e-6)


@pytest.mark.parametrize("X,Y,expected", [
    ([(-1, -1, 0), (1, -1, 0), (0, 1, 0)], [(0, -0.5, -1), (0, -0.5, 1), (0, 0.5, 0)], 1),   # transversal
    ([(0, 0, 0), (4, 0, 0), (0, 4, 0)], [(0, 0, 0), (1, 0, 0), (0, 1, 0)], 1),             # coplanar, nested
    ([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 0, 0), (1, 0, 0), (0, 0, 1)], 0),             # shared edge only
    ([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(2, 2, 0), (3, 2, 0), (2, 3, 0)], 0),             # coplanar, disjoint
    ([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 0, 0), (-1, 0, 1), (0, -1, 1)], 0),           # shared vertex only
    ([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 0, 1), (1, 0, 1), (0, 1, 1)], 0),             # parallel planes
])
def test_intersection_check_cases(X, Y, expected):
    P = np.array(X + Y, float)
    assert len(gn.intersecting_triangle_pairs(P, np.array([[0, 1, 2], [3, 4, 5]]), np.array([0, 1]))) == expected


def test_uniform_mesh_has_the_documented_post_overlap():               # the overlap exists and is detected
    m = gn.GN(3, 10.0, 15.0).surface_mesh(12, 16)
    assert len(gn.intersecting_triangle_pairs(m.points, m.cells_dict["triangle"], m.cell_data["leaflet"][0])) > 0
