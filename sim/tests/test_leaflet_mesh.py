"""Checks of the G1 fillet construction (C-019) and of the thin-solid meshes used in A5 (C-020)."""

import numpy as np
import pytest

from wheatley import leaflet_mesh as lm


@pytest.mark.parametrize("z", [0.2, 0.5, 0.8, 1.0])
def test_fillet_is_tangent_to_both_circles(z):
    rho = 0.1 * z
    t1, t2, F, T1, T2, p0, p1 = lm.fillet(z, rho)
    c1, R1, c2, R2, Rm = lm._circles(z)
    # tangency: centre at distance R -+ rho, tangency points on both circles and on the fillet
    assert np.isclose(abs(np.linalg.norm(F - c1) - R1), rho)
    assert np.isclose(abs(np.linalg.norm(F - c2) - R2), rho)
    assert np.isclose(np.linalg.norm(T1 - F), rho) and np.isclose(np.linalg.norm(T2 - F), rho)
    assert np.pi <= t1 < lm.t_M(z) and np.pi <= t2 < lm.t_M(z)


@pytest.mark.parametrize("z", [0.3, 0.7, 1.0])
def test_filleted_section_is_tangent_continuous(z):
    S = lm.section(z, 400, 400, 200, 0.1 * z)
    d = np.diff(S, axis=0)
    d = d[np.linalg.norm(d, axis=1) > 1e-12]
    turn = np.degrees(np.abs(np.diff(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))))
    assert turn.max() < 1.0                        # no kink left (fine sampling)
    S0 = lm.section(z, 400, 400, 0, 0.0)
    d0 = np.diff(S0, axis=0)
    d0 = d0[np.linalg.norm(d0, axis=1) > 1e-12]
    turn0 = np.degrees(np.abs(np.diff(np.unwrap(np.arctan2(d0[:, 1], d0[:, 0])))))
    assert turn0.max() > 30.0                      # the sharp section has an in-plane kink at M


def test_fillet_strip_not_developable():                               # C-019 (with C-018)
    K = lm.fillet_strip_gaussian_curvature(0.1, nz=8)
    assert np.abs(K[:, 2]).max() > 1.0


@pytest.mark.parametrize("rho0", [0.0, 0.1])
def test_solid_mesh_is_conforming_and_positive(rho0):
    G = lm.surface_grid(24, 33, 20, 8, rho0=rho0, rho_min=0.03)       # corner cut (the uncut
    skip, _ = lm.corner_skip(G, 33, 0.125)                              # apex used to pass only because inverted
    X, tris, idx = lm.triangulate(G, skip=skip)                         # tets were silently flipped)
    Y, tets, n, layer = lm.thin_solid(X, tris, thickness=0.3, layers=2)
    p = Y[tets]
    vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    assert (vol > 0).all()
    faces = np.sort(np.vstack([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]],
                               tets[:, [1, 2, 3]]]), axis=1)
    _, cnt = np.unique(faces, axis=0, return_counts=True)
    assert cnt.max() <= 2                          # every face shared by at most two tets
    # boundary faces = 2 * surface triangles (inner/outer) + side faces; interior faces are shared
    n_boundary = int((cnt == 1).sum())
    assert n_boundary >= 2 * len(tris)


@pytest.mark.parametrize("rho0", [0.0, 0.10])
def test_thin_solid_is_valid_with_corner_cut(rho0):                    # no twisted prisms, no overlaps
    R, H, T = 10.0, 1.5, 0.3
    G = lm.surface_grid(24, 33, 20, 8, rho0=rho0, radius=R, h=H, rho_min=T / R)
    skip, kcut = lm.corner_skip(G, 33, 0.125)
    assert kcut == 3
    X, tris, _ = lm.triangulate(G, skip=skip)
    Y, tets, _, _ = lm.thin_solid(X, tris, thickness=T, layers=2)     # raises if invalid
    p = Y[tets]
    vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    assert np.all(vol > 0)
    assert lm.count_same_side_faces(Y, tets) == 0
    assert lm.count_overlapping_tets(Y, tets)[0] == 0


def test_global_overlap_check_rejects_coarse_mesh():                    # witness (tets 3074, 3211)
    G = lm.surface_grid(16, 22, 13, 6, rho0=0.0, rho_min=0.03)
    skip, _ = lm.corner_skip(G, 22, 0.125)
    X, tris, _ = lm.triangulate(G, skip=skip)
    Y, tets, _, _ = lm.thin_solid(X, tris, thickness=0.3, layers=2, global_check=False)
    n, _ = lm.count_overlapping_tets(Y, tets)
    assert n > 0
    with pytest.raises(ValueError):
        lm.thin_solid(X, tris, thickness=0.3, layers=2)


def test_thin_solid_rejects_the_uncut_apex():                           # the defect that used to be hidden
    G = lm.surface_grid(12, 16, 10, 4, rho0=0.0, radius=10.0, h=1.5)
    X, tris, _ = lm.triangulate(G)
    with pytest.raises(ValueError):
        lm.thin_solid(X, tris, thickness=0.3, layers=2)


def test_corner_skip_requires_integer_rows():
    G = lm.surface_grid(10, 16, 10, 4, rho0=0.0, radius=10.0, h=1.5)
    with pytest.raises(ValueError):
        lm.corner_skip(G, 16, 0.125)
