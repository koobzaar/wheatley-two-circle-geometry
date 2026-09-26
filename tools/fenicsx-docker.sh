#!/usr/bin/env bash
# Run a command in the official FEniCSx image (PETSc + MPI + SLEPc), repo mounted at /work.
# Usage: tools/fenicsx-docker.sh python3 sim/fem/smoke_poisson.py
#        tools/fenicsx-docker.sh mpirun -n 12 python3 sim/fem/<script>.py
# Requires Docker Desktop running. Image: dolfinx/dolfinx:stable.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -W 2>/dev/null || pwd)"
export MSYS_NO_PATHCONV=1
exec docker run --rm --init --shm-size=4g -v "$ROOT:/work" -w /work \
  -e OMP_NUM_THREADS=1 dolfinx/dolfinx:stable "$@"
