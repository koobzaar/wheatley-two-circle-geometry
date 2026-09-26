# G_N — the N-leaflet two-circle family, ready to use

This package turns the two-circle construction of McKee, Oliveira, Cuminato *et al.* (2026) into an explicit family
G_N of N-leaflet valves, with exact formulas for points, derivatives, normals and first fundamental form, the location
and angle of the crease, and surface meshes. It is meant as input for a study that compares different numbers of
leaflets (N = 2, 3, 4, …).

**Status.** The formulas below are the ones stated and checked in the accompanying technical note (claims in
`CLAIMS.md`); the algebraic identities are machine-checked in Lean 4, and the code is tested against them.
The N-leaflet generalisation is ours: it keeps the two tangent principal circles and rotates the second one by 2π/N.
It need not coincide with the family G_N(α) of the talk slides, which we could not reconstruct.

## The family

N an integer ≥ 2, ring radius R > 0, height H > 0, θ = 2π/N, section parameter z ∈ [0, 1] (b = z/2, a = 1 − b; the paper's linear profile).
Leaflet 0 has two pieces over Ω = {(t, z) : 0 ≤ z ≤ 1, π ≤ t ≤ t_M(z)}:

- piece 1: X₁(t, z) = ( R(−b + a cos t), R a sin t, H z )
- piece 2: X₂(t, z) = Rot_θ ( R(−a + b cos t), R b sin t, H z )

with the common junction parameter t_M(z) = 2π − arccos((2ab − (a² + b²) cos θ)/Δ), Δ = a² + b² − 2ab cos θ
(t_M(0) = π + θ, t_M(1) = 2π). Leaflet k is Rot_{kθ} of leaflet 0. The free edges close for every N (at z = 1, piece 2
of leaflet k coincides with piece 1 of leaflet k+1), and the top-view projection of the N leaflets has area πR² for
every N, and the projections cover the disk with multiplicity one almost everywhere (proved by hand in the note,
Proposition 7: Stokes' theorem and the area formula; not formalised).

**Crease.** The two pieces meet along M(z) at a fold angle δ(z) with
cos δ = ((1 + cos t_M)² − 4h² cos θ) / (4h² + (1 + cos t_M)²), h = H/R. There is no crease for N = 2; for N ≥ 3 the
normal jumps by δ along the whole junction. Exact normals are provided on each side; a CFD mesh should keep the
junction as an edge (the exported meshes are conforming along it and flag its vertices).

**Singular point.** The apex of piece 2 at z = 0 (the base of each commissure post) has no tangent plane.

## Comparing different N (criterion)

Keep the **total leaflet area** fixed across N, leave the
opening area as a response, and fix either the ring radius or the height.

- `fixed_R`: R fixed, H chosen so that the total area equals S;
- `fixed_H`: H fixed, R chosen so that the total area equals S;
- `paper`: R and H given.

The total area scales as R²·A_tot_N(H/R) (A_tot_N = N·A_N, A_N the area of one leaflet), so every dimensionless quantity depends only on (N, h = H/R). The two criteria
select **different** shapes. Keeping R and H fixed while matching the area requires changing the vertical profile, and
is impossible for N = 4 (certified in interval arithmetic) and for N = 5, 6, 7 (lower bound) when H = R, with the N = 3 linear
design as reference; with H = 1.5 R it is possible for N = 4
(see the note; "impossible" is over all C¹ non-decreasing vertical profiles g with g(0) = 0, g(1) = 1). Tables for the
reference design (N = 3, R = 10 mm, H = 15 mm) are in `table_fixed_R.csv` and
`table_fixed_H.csv`.

## Use

```python
from wheatley.gn import family
S = family(3, "paper", R=10.0, H=15.0).area()          # reference: N = 3 with the dimensions of the paper's models
G = family(4, "fixed_R", R=10.0, S=S)                  # N = 4 with the same total area and ring radius
G.H, G.h                                               # height (mm) and aspect ratio
G.X1(t, z); G.X2(t, z); G.X(piece, t, z, k)           # points (arrays broadcast)
G.derivatives(piece, t, z)                             # exact (X_t, X_z)
G.normal(piece, t, z, k); G.first_form(piece, t, z)   # unit normal (consistent orientation), (E, F, G)
G.junction(z); G.fold_angle_deg(z)                     # crease curve and angle
G.surface_mesh(nz=40, nt=60).write("G4.vtk")           # meshio mesh, see "Mesh contract" below
```

Scripts: `sim/scripts/gn_export.py` writes the tables and the VTK/STL meshes for N = 2…7 and both criteria.
Tests: `sim/tests/test_gn.py` (derivatives and normals against finite differences, junction and fold angle, closure of
the free edges, area scaling, the equal-area families, mesh conformity).

## Mesh contract

- Each leaflet is a separate surface. Where leaflets touch (the commissure posts, and the free edges of neighbours at
  z = 1) the geometry coincides but the vertices are **not** merged. The surface is not watertight and is not a
  volume boundary; merge or offset it as your solver requires.
- Within a leaflet, the two pieces share the junction vertices (conforming along the crease). Triangles follow the
  leaflet's consistent orientation, so the two faces at a crease edge traverse it in opposite directions.
- Normals: cell data `normal` (exact normal of the piece), point data `normal1` and `normal2` (exact normal of piece 1 /
  piece 2; both are present on the crease, where they differ by the fold angle; `normal2` is undefined at the apex of
  piece 2). Point data `on_crease` flags the junction; cell data `leaflet` and `piece` label the triangles.
- Near the apex of piece 2 (the base of each commissure post) the uniform grid in (t, z) produces thin triangles: the minimum
  angle is about 0.5° at 40×60 and decreases with refinement (the apex is a cone vertex). For solvers that need shape-regular
  elements, remesh from the exact parametrisation (`G.X`, `G.derivatives`) with a mesher of your choice.
- Contact between leaflets: the exact surfaces of neighbouring leaflets are tangent along each commissure post and
  coincide along the free edges at z = 1. In the exported meshes the free edges of neighbours coincide exactly (same
  vertices positions), but along the posts the first strip of triangles of the two leaflets is coplanar and nested: all
  intersections between different leaflets are these overlaps. Every point of an overlapping triangle lies within one element
  length (the largest edge) of a post line: the largest distance of a vertex of an overlapping triangle to the nearest post is
  1.30 / 0.35 / 0.17 mm at 12×16 / 40×60 / 80×120 for N = 3 (R = 10 mm, H = 15 mm) and 1.05 mm at 20×30 for N = 2 (the distance
  to a vertical line is convex, so its maximum over a triangle is at a vertex). This only locates the overlaps; it does not
  bound how much two triangles overlap. Checked by `intersecting_triangle_pairs` in the tests. A solver that cannot accept touching or overlapping boundaries
  must merge or offset the leaflets there.
- Use VTK (or another format that keeps data). The STL files contain the geometry only and lose every label and normal.

## Limitations

- Only the paper's linear vertical profile; other profiles change the area and the crease (see the note).
- Geometry only: no material, thickness or loads. The mechanical check in the note is separate and preliminary.
- The family is our generalisation of the 2026 construction, not the harmonic family G_N(α) of the slides.
