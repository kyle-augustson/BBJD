# BBJD — Block-Bordered Jacobi-Davidson Eigensolver

Sparse-matrix eigensolver for generalized non-Hermitian problems
`A x = λ B x`, targeting pencils with block-banded + tau-row structure
(typical of Dedalus-style spectral PDE discretizations).

Returns `k` eigenvalues near a user-specified shift `σ`. Uses the
JDQZ variant (Fokkema-Sleijpen-van der Vorst 1999) with Schur-vector
locking and preconditioned Sleijpen-van der Vorst correction.

## Build

```bash
cd /path/to/bbjd
python setup.py build_ext --inplace
```

Requires a C11 compiler with OpenMP, NumPy ≥ 1.22, and SciPy ≥ 1.9.
Links against SciPy's bundled OpenBLAS by default; for production
deployment on AMD Epyc, build your own OpenBLAS or AOCL/BLIS with
`MAX_THREADS=128` (or whatever matches your target thread count per
MPI rank) and point `setup.py` at it.

## Test

```bash
python tests/test_bbjd.py
```

Runs 20 tests covering correctness vs `scipy.sparse.linalg.eigs`,
edge cases, API contracts, and regressions for specific bugs fixed
during development.

## Basic usage

```python
import scipy.sparse as sp
from spdee.solvers.bbjd import jdqz
from spdee.solvers.fplus import SparseLU

# A, B are scipy sparse matrices (CSR preferred)
sigma = 3.0 + 0j
M = (A - sigma * B).tocsr()
lu = SparseLU(M, colamd=True, pin_dense_rows=True)

evals, evecs, info = jdqz(
    A, B,
    k=64,                   # number of eigenvalues to find
    sigma=sigma,
    tol=1e-10,
    preconditioner=lu,
    maxsub=200, minsub=80,  # subspace dimensions
    maxiter=2000,
    return_info=True,
)
print(f"Converged {info['nconv']}/{64} in {info['iters']} iterations")
```

## Deployment (PDE eigenspectrum sweep)

Each MPI rank runs `jdqz` independently on one guess eigenvalue σ.
The LU solve (called every JD iteration) and BBJD's BLAS calls
share the thread pool within a rank — they never run concurrently,
so no oversubscription risk.

### Launcher

```bash
# 4 ranks per node × 32 threads per rank on Epyc 9745 (128 cores/node)
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export OMP_PROC_BIND=close
export OMP_PLACES=cores

mpirun -np $NRANKS \
    numactl --cpunodebind=$((LOCAL_RANK / RANKS_PER_NUMA)) \
            --membind=$((LOCAL_RANK / RANKS_PER_NUMA)) \
    python eigensweep.py
```

`numactl` pins each rank to one NUMA node. BBJD's workspace
allocator uses first-touch initialization so the large arrays
(V, W, AV, BV) end up on the rank's local NUMA node.

### Memory per rank

At n = 10^6, nev = 64, maxsub = 200:

| Array | Size |
|-------|------|
| V, W, AV, BV, swap_nm | 3.2 GB each × 5 = 16 GB |
| Q_right, Q_left | 1.0 GB combined |
| gmres_V (inner_maxiter=50; 16 MB when inner_maxiter=0) | 0.8 GB |
| Misc (tmps, projected pencil, LAPACK work) | < 1 GB |
| **BBJD total** | **~18 GB** |

Add whatever your matvec callback and LU factors need on top.

## Algorithm notes

- **Correction equation**: with `inner_maxiter=0` (default) the
  Sleijpen-van der Vorst preconditioned single step
  `t = M^{-1}r - M^{-1}u (u^H M^{-1}r) / (u^H M^{-1}u)`, the exact
  correction when `M` is the factorization of `A - sigma B`; the outer
  subspace iteration then does the search an inner solve would
  duplicate.  With `inner_maxiter > 0` and an approximate `M` (a
  domain-decomposition or multigrid preconditioner), the projected
  correction equation `(I - Q_l Q_l^H)(beta A - alpha B)(I - Q_r Q_r^H) t = -r`
  is solved by left-preconditioned GMRES to `inner_tol`, with `Q_r` the
  locked eigenvectors and the Ritz vector and `Q_l` their test images.

- **Starting vector**: `v0` seeds the search space (an eigenvector from
  a coarser discretization or a nearby parameter); random otherwise.

- **Locking**: Schur-vector style via ZGGES + ZTGSEN. After
  convergence, `nlock` increments but V is not shrunk — the Schur
  form keeps locked eigenvalues in the top-left block.

- **Restart**: thick, preserving the Schur structure. V, AV, BV
  are compressed via ZB (Schur right-transform) to keep the best
  `minsub` Schur vectors.

- **Numerical stability**: DGKS-monitored CGS2 for orthogonalization
  (BLAS-2 zgemv), `zlartg`-style Givens rotations with `cs` real
  and non-negative (LAPACK convention).

## Files

- `include/bbjd.h` — public API and workspace struct
- `include/bbjd_blas.h` — BLAS/LAPACK prototypes with scipy prefix handling
- `src/jd_core.c` — main JDQZ iteration, correction, restart
- `src/_bbjd_c.c` — Python C extension, callback trampolines
- `bbjd.py` — scipy-`eigs`-style Python wrapper
- `setup.py` — build script (links against scipy's OpenBLAS)
- `tests/test_bbjd.py` — pytest-compatible test suite

## License

GPL-3.0-or-later (see `LICENSE`).  This copy is vendored into SPDEE; the
library is developed in its own repository.
