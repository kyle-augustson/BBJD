# bbjd in spdee — status notes

The C core is the April 2026 JDQZ (`src/jd_core.c`) with one portability
fix in `include/bbjd_blas.h`: `zdotc_` is routed through `cblas_zdotc_sub`.
On an MKL-based NumPy the Fortran `zdotc_` symbol resolves to MKL's copy,
which returns complex values through a hidden pointer instead of in
registers, and the solver segfaulted on its first dot product. The CBLAS
`_sub` form has one C ABI in every implementation.

## The one-eigenvalue stall, diagnosed and fixed (2026-09-17)

On the pencils spdee produces the solver used to converge the eigenvalue
nearest the shift and then stall with its residual pinned at 1e-2 while
the Ritz value sat on an exact eigenvalue. Two defects, found with a
numpy reference JDQZ on the same pencils (search space expanded by the
Sleijpen-van der Vorst step with the exact factorization, test space
and extraction varied):

1. **Test space.** `W = orth(B V)` (a Galerkin projection) places spurious
   Ritz values near an interior target — at m = 0 the test shift
   0.02 + 0.3i sits in a gap, the nearest eigenvalue 0.27 away, and the
   projected pencil put values within 0.05 of it every iteration — and
   its columns degenerate on the null rows of a singular B. The harmonic
   choice of JDQZ (Fokkema, Sleijpen & van der Vorst 1998),
   `W = orth(nu0 A V + mu0 B V)` with `(nu0, mu0) = (1, -sigma)/sqrt(1 + |sigma|^2)`
   for the eigenvalues nearest sigma (sigma = 0 for smallest magnitude,
   `W = orth(B V)` for the other targets), is used at all three sites:
   the start vector, the expansion and the restart.
2. **Ritz vector.** The core took `u = V ZB[:, p]`, the right generalized
   Schur vector of the projected pencil, which is an eigenvector only at
   p = 0. After the first lock the target sits at p > 0 and its Schur
   vector couples to the locked directions, so the residual stalls at
   the size of that coupling; JDQZ would remove it with the deflation
   projectors the core built (`Q_left`, `Q_right`) but never applied.
   The eigenvector is now recovered from the triangular pair by
   back-substitution, `(beta S_A - alpha S_B) y = 0` with `y[p] = 1`,
   and `u = V ZB y`; the projector construction is gone.

A third change makes the thick restart keep what it should: the reordering
selects the `minsub` eigenvalues nearest the target (the locked ones and
the target among them) rather than the locked pair and the target alone,
so the leading block the restart keeps is the nearest Schur vectors.

Results with the exact factorization of `K - sigma M` as preconditioner,
`tol = 1e-10`, `maxsub = 60`: the m = 2 collocated inertial pencil
(n = 800, 320 zero rows in M) 6/6 in 121 iterations, the m = 0 pencil
(280 zero rows and the dense integral gauge row) 6/6 in 116, both sets
those of ARPACK to 1e-10; `tests/test_bbjd.py` 21/21 in 0.7 s (three of
them had been failing at 30 s). `tests/test_solvers.py::test_bbjd` and
`test_bbjd_gauge_row` cover the two pencils.

`bbjd_correction_gmres` (left-preconditioned GMRES on the projected
correction equation) runs when `inner_maxiter > 0`, for an approximate
preconditioner; the deflation projectors are the orthonormalized locked
eigenvectors plus the Ritz vector and their test images, appended at each
lock and rebuilt for the Ritz vector at each correction.  With
`inner_maxiter = 0` (the default) the single Sleijpen-van der Vorst step
with the exact factorization is the correction.  `v0` seeds the search
space.
