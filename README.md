# Geometry of the two-circle Wheatley leaflet: proofs and code

This repository supports the technical note

> B. B. Trigueiro, *On the Geometry of the Two-Circle Wheatley Leaflet*, technical note, Instituto de Ciências Matemáticas
> e de Computação (ICMC), Universidade de São Paulo (USP), September 2026.
> [`notes/technote/main.pdf`](notes/technote/main.pdf) (English) · [`notes/technote/nota_ptbr.pdf`](notes/technote/nota_ptbr.pdf) (Portuguese)

The note studies the leaflet geometry of the two-circle construction of S. McKee, H. L. Oliveira, J. A. Cuminato *et al.*,
*Two Circle Construction of the Reinforced Wheatley Valve*, Biomedical Materials & Devices (2026),
doi:[10.1007/s44174-026-00691-4](https://doi.org/10.1007/s44174-026-00691-4), and its extension to N leaflets.

Every labelled statement `[C-xxx]` of the note is listed in [`CLAIMS.md`](CLAIMS.md) with its status (proved in Lean,
verified numerically, or argued by hand) and the files that support it.

## Contents

| Path | What it is |
|------|------------|
| `lean/` | Lean 4 + mathlib proofs of the algebraic identities (`WheatleyGeom/*.lean`) |
| `sim/src/wheatley/` | Python package: the parametrisations, flat patterns, N-leaflet family, G_N package, meshes |
| `sim/tests/` | Automated tests (pytest) of every numerical statement |
| `sim/scripts/` | Scripts that produce the numbers, tables and figures of the note |
| `sim/fem/` | The exploratory finite-element check of Section 7 (FEniCSx) |
| `sim/out/` | Small outputs quoted in the note (interval certificate, equal-area family, FEM summary) |
| `deliverable/gn/` | The G_N package: README and tables of the equal-area families |
| `notes/technote/` | The note (LaTeX sources, figures, PDFs) |

## Checking the proofs (Lean)

Requires [elan](https://github.com/leanprover/elan). The toolchain is pinned in `lean/lean-toolchain` and mathlib in
`lean/lake-manifest.json` (and `lean/lakefile.toml`).

```bash
cd lean
lake exe cache get      # download the compiled mathlib
lake build              # checks every proof
```

The files contain no `sorry`, `axiom` or `native_decide`; the theorems use only the standard axioms (`propext`,
`Classical.choice`, `Quot.sound`), which `#print axioms <name>` confirms. What remains to be trusted is that each Lean
statement says what the note says: the Lean names are given next to each statement of the note and in `CLAIMS.md`.

## Running the numerical checks (Python)

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12).

```bash
uv run --project sim pytest -q                                   # all tests
uv run --project sim python sim/scripts/technote_proof_checks.py # symbolic checks of every step of the written proofs
uv run --project sim python sim/scripts/technote_figures.py      # figure data and 3D renders (drawn by LaTeX/TikZ)
uv run --project sim python sim/scripts/a6_certificate_n4.py     # interval certificate (Section 6)
uv run --project sim python sim/scripts/a6_equal_area_family.py  # Table 1
uv run --project sim python sim/scripts/gn_export.py             # G_N tables and VTK/STL meshes
```

Other scripts are listed in `CLAIMS.md` next to the statements they support; each has its usage in its docstring.

## The finite-element check (optional)

Section 7 of the note is exploratory. It uses FEniCSx through Docker (`dolfinx/dolfinx:stable`):

```bash
uv run --project sim python sim/scripts/make_leaflet_meshes.py            # meshes (see docstring for the variants)
tools/fenicsx-docker.sh mpirun -n 12 python3 sim/fem/leaflet_pressure.py crease
tools/fenicsx-docker.sh mpirun -n 12 python3 sim/fem/leaflet_pressure.py fillet
uv run --project sim python sim/scripts/fem_summary.py && uv run --project sim python sim/scripts/fem_compare.py
```

## The G_N package

`sim/src/wheatley/gn.py` gives the N-leaflet family with exact normals, first fundamental form, the crease location and
fold angle, and the equal-area criteria of Section 6; see [`deliverable/gn/README.md`](deliverable/gn/README.md).

## License

MIT (see `LICENSE`). If you use this material, please cite the note and the paper above.
