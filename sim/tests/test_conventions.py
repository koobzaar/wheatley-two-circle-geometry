"""Internal-consistency checks of the transcription in notes/technote/main.tex (Section 1) (McKee et al. 2026).

Each test names the ambiguity (A-n) or equation it backs.  Formulas are written here *directly
from the paper's equations* (not imported from wheatley.geometry) so that they cross-check the
package; the last tests compare the package against them.
"""

import numpy as np
import pytest

from wheatley import geometry as g

R3 = np.sqrt(3.0)
B = np.linspace(0.0, 0.5, 26)          # b = z/2 over the closed domain z in [0, 1]
T = np.linspace(-np.pi, 3 * np.pi, 41)


def rot_ccw(theta, x, y):
    """R_theta, counter-clockwise (the convention adopted in CONVENTIONS.md, section 2)."""
    return np.cos(theta) * x - np.sin(theta) * y, np.sin(theta) * x + np.cos(theta) * y


def map34(theta, x, y):
    """Eqs. (3)-(4) read as a point map (x, y) -> (x', y')."""
    return np.cos(theta) * x + np.sin(theta) * y, -np.sin(theta) * x + np.cos(theta) * y


def circ1(a, x, y):   # eq. (1), = 0 on the circle
    return (x + 1 - a) ** 2 + y ** 2 - a ** 2


def circ2(b, x, y):   # eq. (2)
    return (x + 1 - b) ** 2 + y ** 2 - b ** 2


def D(b):
    return 1 - b + b * b


# --- A-1: sense of rotation in (3)-(4) --------------------------------------------------

@pytest.mark.parametrize("b", B)
def test_A1_eqs17_18_are_ccw_rotation_of_13_14(b):
    a = 1 - b
    x13, y14 = -(1 - a) + a * np.cos(T), a * np.sin(T)
    x17 = 0.5 * ((1 - a) - a * (np.cos(T) + R3 * np.sin(T)))
    y18 = 0.5 * (-R3 * (1 - a) + a * (R3 * np.cos(T) - np.sin(T)))
    assert np.allclose(rot_ccw(2 * np.pi / 3, x13, y14), (x17, y18))
    x21 = 0.5 * ((1 - a) - a * (np.cos(T) - R3 * np.sin(T)))
    y22 = 0.5 * (R3 * (1 - a) - a * (R3 * np.cos(T) + np.sin(T)))
    assert np.allclose(rot_ccw(4 * np.pi / 3, x13, y14), (x21, y22))


@pytest.mark.parametrize("b", B)
def test_A1_eqs19_20_lie_on_circle8_defined_by_substitution(b):
    # (8) is (2) with (x, y) replaced by map34(2pi/3, x, y); (19)-(20) must satisfy it.
    x19 = 0.5 * ((1 - b) - b * (np.cos(T) + R3 * np.sin(T)))
    y20 = 0.5 * (-R3 * (1 - b) + b * (R3 * np.cos(T) - np.sin(T)))
    xp, yp = map34(2 * np.pi / 3, x19, y20)
    assert np.allclose(circ2(b, xp, yp), 0, atol=1e-12)


def test_A1_map34_is_clockwise_as_point_map():
    # (3)-(4) as a point map sends E=(-1,0) to (1/2, +sqrt3/2): clockwise by 2pi/3 ...
    assert np.allclose(map34(2 * np.pi / 3, -1.0, 0.0), (0.5, R3 / 2))
    # ... while the circle obtained by substitution is centred on the CCW image of B.
    assert np.allclose(rot_ccw(2 * np.pi / 3, -1.0, 0.0), (0.5, -R3 / 2))


# --- A-2: apex of surface 2 ---------------------------------------------------------------

def test_A2_surface2_at_z0_is_P_not_printed_apex():
    t = np.linspace(np.pi, 5 * np.pi / 3, 9)
    b = 0.0
    x = 0.5 * ((1 - b) - b * (np.cos(t) + R3 * np.sin(t)))          # (36)
    y = 0.5 * (-R3 * (1 - b) + b * (R3 * np.cos(t) - np.sin(t)))    # (37)
    assert np.allclose(x, 0.5) and np.allclose(y, -R3 / 2)
    # printed apex of Prop. 1 = (3)-(4) point map applied to (-1, 0, 0) of eq. (53)
    assert np.allclose(map34(2 * np.pi / 3, -1.0, 0.0), (0.5, R3 / 2))


def test_A2_six_apex_set_unchanged():
    printed = {(-1.0, 0.0, 2.0), (0.5, R3 / 2, 0.0), (0.5, R3 / 2, 2.0), (0.5, -R3 / 2, 0.0),
               (0.5, -R3 / 2, 2.0), (-1.0, 0.0, 0.0)}                  # Prop. 1 / (54)
    ours = set()
    for k in range(3):
        for (x, y, z) in [(-1.0, 0.0, 2.0), (0.5, -R3 / 2, 0.0)]:
            xr, yr = rot_ccw(2 * np.pi * k / 3, x, y)
            ours.add((xr, yr, z))
    key = lambda s: sorted(tuple(np.round(p, 12)) for p in s)
    assert key(printed) == key(ours)


# --- A-4 / A-5: eqs. (27)-(29) --------------------------------------------------------------

@pytest.mark.parametrize("b", B)
def test_A4_eq28_needs_sqrt3_factor(b):
    a = 1 - b
    Q = a * a + a * b + b * b
    c27 = -(2 * Q - 3 * (a + b)) / (2 * Q)
    s_fixed = -R3 * (a - b) / (2 * Q)
    assert np.isclose(c27 ** 2 + s_fixed ** 2, 1.0)
    # printed (28): the residual on a = 1 - b is -(2b-1)(3b-1)/(2 D^2) (audit_eqs_27_29.py), so the printed form
    # fails as an *identity* of the family but happens to hold at the isolated values b = 1/3 and b = 1/2.
    s_printed = -np.sqrt(a - b) / (2 * Q)
    resid = c27 ** 2 + s_printed ** 2 - 1.0
    assert np.isclose(resid, -(2 * b - 1) * (3 * b - 1) / (2 * D(b) ** 2))


def test_A4_printed_eq28_fails_as_identity_but_has_isolated_zeros():
    def resid(b):
        a = 1 - b
        Q = a * a + a * b + b * b
        return (-(2 * Q - 3 * (a + b)) / (2 * Q)) ** 2 + (np.sqrt(a - b) / (2 * Q)) ** 2 - 1.0
    assert np.isclose(resid(0.25), -16 / 169)
    assert np.isclose(resid(1 / 3), 0.0) and np.isclose(resid(0.5), 0.0)


# --- eqs. (30)-(35): L, M, G, H ---------------------------------------------------------------

@pytest.mark.parametrize("b", B)
def test_LM_on_circles_30_31_and_GH_mirror(b):
    xL, yL = (1 - 3 * b) / 2, -R3 / 2 * (1 - b)                                   # (32)
    xM, yM = (1 - b - 2 * b * b) / (2 * D(b)), -R3 / 2 * (1 - b) * (1 - 2 * b) / D(b)  # (33)
    for x, y in [(xL, yL), (xM, yM)]:
        assert np.isclose((x + b) ** 2 + y ** 2, (1 - b) ** 2)                    # (30)
        xp, yp = map34(2 * np.pi / 3, x, y)
        assert np.isclose(circ2(b, xp, yp), 0, atol=1e-12)                       # (31)
        # G, H (34)-(35) are the mirror images y -> -y; they lie on (1) and on (12)
        xq, yq = map34(4 * np.pi / 3, x, -y)
        assert np.isclose(circ2(b, xq, yq), 0, atol=1e-12)


# --- arc parameters (pp. 5-8, Appendix A) ----------------------------------------------------

@pytest.mark.parametrize("b", B)
def test_tM_arcsin_and_arccos_forms_agree_and_hit_M(b):
    c = (1 + 2 * b - 2 * b * b) / (2 * D(b))
    s = R3 / 2 * (1 - 2 * b) / D(b)
    t_asin = 2 * np.pi - np.arcsin(s)          # (EIM end, p. 5 / Appendix A plot1)
    t_acos = 2 * np.pi - np.arccos(c)          # (MNP start, p. 6 / Appendix A plot2)
    assert np.isclose(t_asin, t_acos)
    assert 5 * np.pi / 3 - 1e-12 <= t_asin <= 2 * np.pi + 1e-12
    xM, yM = (1 - b - 2 * b * b) / (2 * D(b)), -R3 / 2 * (1 - b) * (1 - 2 * b) / D(b)
    assert np.allclose((-b + (1 - b) * np.cos(t_asin), (1 - b) * np.sin(t_asin)), (xM, yM))       # (38)-(39)
    x40 = 0.5 * ((1 - b) - b * (np.cos(t_acos) + R3 * np.sin(t_acos)))
    y41 = 0.5 * (-R3 * (1 - b) + b * (R3 * np.cos(t_acos) - np.sin(t_acos)))
    assert np.allclose((x40, y41), (xM, yM))                                                        # (40)-(41)


@pytest.mark.parametrize("b", B)
def test_L_based_arc_parameters(b):
    # EIL: t in [pi, 5pi/3] on (1); LNP: t from pi/3 down to -pi on (8)=(36)-(37)
    xL, yL = (1 - 3 * b) / 2, -R3 / 2 * (1 - b)
    t = 5 * np.pi / 3
    assert np.allclose((-b + (1 - b) * np.cos(t), (1 - b) * np.sin(t)), (xL, yL))
    for t, target in [(np.pi / 3, (xL, yL)), (-np.pi, (0.5, -R3 / 2))]:
        x = 0.5 * ((1 - b) - b * (np.cos(t) + R3 * np.sin(t)))
        y = 0.5 * (-R3 * (1 - b) + b * (R3 * np.cos(t) - np.sin(t)))
        assert np.allclose((x, y), target)


@pytest.mark.parametrize("b", B)
def test_counter_rotational_arcs_are_mirror_of_correct_ones(b):
    # (23)-(24) = mirror (y -> -y) of (19)-(20) with t -> -t; EFH/HJK ranges mirror EIM/MNP (p. 8)
    x19 = 0.5 * ((1 - b) - b * (np.cos(-T) + R3 * np.sin(-T)))
    y20 = 0.5 * (-R3 * (1 - b) + b * (R3 * np.cos(-T) - np.sin(-T)))
    x23 = 0.5 * (1 - b - b * (np.cos(T) - R3 * np.sin(T)))
    y24 = 0.5 * (R3 * (1 - b) - b * (R3 * np.cos(T) + np.sin(T)))
    assert np.allclose((x19, -y20), (x23, y24))
    theta = np.arcsin(R3 / 2 * (1 - 2 * b) / D(b))                       # EFH end (p. 8)
    assert np.isclose(theta, np.arccos(0.5 * (1 + 2 * b - 2 * b * b) / D(b)))   # HJK start (p. 8)
    assert np.isclose(theta, 2 * np.pi - g.t_M(2 * b))


# --- package vs paper ---------------------------------------------------------------------------

@pytest.mark.parametrize("z", np.linspace(0, 1, 11))
def test_package_matches_eqs_38_41_and_appendixA(z):
    b = z / 2
    t = np.linspace(np.pi, g.t_M(z), 13)
    P1 = g.surface1(t, z)
    assert np.allclose(P1[:, 0], -b + (1 - b) * np.cos(t)) and np.allclose(P1[:, 1], (1 - b) * np.sin(t))
    assert np.allclose(P1[:, 2], 2 * b)                                  # Maple: third coordinate 2*b
    P2 = g.surface2(t, z)
    assert np.allclose(P2[:, 0], 0.5 * ((1 - b) - b * (np.cos(t) + R3 * np.sin(t))))
    assert np.allclose(P2[:, 1], 0.5 * (-R3 * (1 - b) + b * (R3 * np.cos(t) - np.sin(t))))
    small = rot_ccw(2 * np.pi / 3, -(1 - b) + b * np.cos(t), b * np.sin(t))   # R_{2pi/3} of (15)-(16)
    assert np.allclose(P2[:, :2], np.stack(small, axis=-1))


@pytest.mark.parametrize("k", [0, 1, 2])
def test_leaflet_grid_closed_domain_includes_base_arc_and_apex(k):
    # K-4: with z_min=0 the grid covers the closed Omega, including z = 0
    P1, P2 = g.leaflet_grid(nt=9, nz=5, k=k, z_min=0.0)
    assert np.allclose(P1[0, :, 2], 0) and np.allclose(np.linalg.norm(P1[0, :, :2], axis=-1), 1)  # base arc E..P
    apex = g.rot_ccw(g.APEX_SURFACE2, 2 * np.pi * k / 3)
    assert np.allclose(P2[0], apex)                                                                # row z=0 = apex
    assert np.allclose(P1[-1, -1], P2[-1, 0])                                                      # meet at M(1)
