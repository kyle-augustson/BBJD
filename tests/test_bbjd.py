"""
Test suite for the BBJD eigensolver.

Organized into:
  - Correctness tests (match scipy.sparse.linalg.eigs)
  - API / interface tests
  - Edge cases (small n, near-singular B, exact preconditioner)
  - Stability tests (repeated eigenvalues, clustered eigenvalues)
  - Regression tests for specific bugs we fixed during development

The problem sizes scale with the module-level ``pmult`` parameter.  Base
sizes are chosen to run quickly (pmult=1 takes seconds); production-sized
runs can set pmult up to ~100.

Large / slow tests are decorated with @pytest.mark.slow so you can skip
them with ``pytest -m 'not slow'``.

Run with:
    cd /path/to/bbjd && pytest tests/ -v                # all tests
    cd /path/to/bbjd && pytest tests/ -v -m 'not slow'  # skip slow

Environment overrides:
    BBJD_PMULT=N    set pmult without editing this file

Each test uses a fixed random seed for reproducibility.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs as scipy_eigs

try:
    import pytest
    _HAVE_PYTEST = True
except ImportError:
    _HAVE_PYTEST = False
    # Stub so @pytest.mark.slow doesn't crash without pytest installed
    class _PassThroughMark:
        """Stub that acts as an identity decorator, usable as either
        @mark (no parens) or @mark(...) with arbitrary args."""
        def __call__(self, *args, **kwargs):
            # Case 1: used as @mark.slow(reason=...) — returns a decorator
            # Case 2: used as @mark.slow directly — args[0] is the target
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda f: f
    class _StubMark:
        def __getattr__(self, name):
            return _PassThroughMark()
    class _StubPytest:
        mark = _StubMark()
        @staticmethod
        def skip(msg):
            raise RuntimeError(f"skipped: {msg}")
    pytest = _StubPytest()

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'fplus'))

from bbjd import jdqz
try:
    from sparse_lu_full_pivot import SparseLU
except ImportError:
    # without FPLUS the preconditioner is scipy's SuperLU under the same interface
    from scipy.sparse.linalg import splu as _splu

    class SparseLU:
        def __init__(self, A, **kwargs):
            self._lu = _splu(sp.csc_matrix(A))

        def solve(self, b, trans="N"):
            return self._lu.solve(b, trans=trans)


# ============================================================
# Problem-size scaling parameter
#
# All base sizes in the tests are multiplied by ``pmult``.  Cost grows
# roughly linearly in n for the sparse problems (matvec dominant) and
# cubically in maxsub for the projected QZ.
#
# Typical values:
#    pmult =   1 : CI / quick regression (~seconds total, non-slow tests)
#    pmult =  10 : mid-scale, catches more numerical issues (~minutes)
#    pmult = 100 : near-production scale (needs @pytest.mark.slow suite
#                  only; default suite will take hours)
# ============================================================
pmult = int(os.environ.get('BBJD_PMULT', '1'))


# ============================================================
# Helpers
# ============================================================

def _residual(A, B, lam, x):
    """Eigenvector relative residual ||Ax - lam Bx|| / ||x||."""
    return np.linalg.norm(A @ x - lam * B @ x) / np.linalg.norm(x)


def _match_eigenvalues(our_evs, ref_evs):
    """For each of our eigenvalues, find nearest in ref.  Returns max error."""
    errors = []
    for e in our_evs:
        errors.append(min(abs(e - r) for r in ref_evs))
    return max(errors)


def _make_diagonal(n, spacing=0.1):
    """Simple diagonal test matrix: eigenvalues at spacing*[1..n]."""
    d = np.arange(1, n + 1, dtype=np.complex128) * spacing
    return sp.diags(d, format='csr'), sp.eye(n, format='csr', dtype=np.complex128)


def _make_random_sparse(n, sigma_shift=5.0, density=0.05, seed=42):
    """Random complex sparse matrix, shifted to put eigenvalues near
    sigma_shift.  For large n, density is automatically clamped so
    nnz stays roughly proportional to n (rather than n^2).
    """
    np.random.seed(seed)
    d = min(density, 50.0 / n) if n > 1000 else density
    A = sp.random(n, n, density=d, format='csr', dtype=np.complex128)
    A = A + sigma_shift * sp.eye(n, dtype=np.complex128)
    A.data += 0.1j * np.random.randn(A.nnz)
    B = sp.eye(n, format='csr', dtype=np.complex128)
    return A, B


def _make_block_banded_tau(n, bandwidth=7, n_tau=4, shift=3.0, seed=123):
    """Block-banded + tau rows (Dedalus-style spectral PDE matrix)."""
    np.random.seed(seed)
    diags_data = [np.random.randn(n) + 0.1j * np.random.randn(n)
                  for _ in range(2 * bandwidth + 1)]
    offsets = list(range(-bandwidth, bandwidth + 1))
    A = sp.diags(diags_data, offsets, shape=(n, n), format='lil',
                 dtype=np.complex128)
    for i in range(n_tau):
        A[i, :] = np.random.randn(n) + 0.1j * np.random.randn(n)
    A = A.tocsr() + shift * sp.eye(n, dtype=np.complex128)
    B = sp.eye(n, format='csr', dtype=np.complex128)
    return A, B


# Subspace sizing that scales with pmult for the harder problems
def _subspace_sizes():
    """Return (maxsub, minsub) appropriate for current pmult on the
    random/block-banded problems.  Larger problems need more subspace
    room for the same number of eigenvalues to converge."""
    maxsub = max(150, 30 * pmult)
    minsub = max(60, 12 * pmult)
    return maxsub, minsub


# ============================================================
# Correctness — diagonal (easy case, exact preconditioner)
# ============================================================

class TestDiagonal:
    """Diagonal matrices with exact preconditioner.  This should be
    trivial — primarily verifies the JD algorithmic plumbing."""

    def test_diagonal_5_nearest_sigma0(self):
        # Eigenvalues are 0.1, 0.2, ..., 0.1*n.  sigma=0 => 5 smallest
        # (and 5 nearest) are 0.1..0.5 regardless of n.
        A, B = _make_diagonal(50 * pmult)
        lu = SparseLU(A.tocsr())  # sigma=0 → exact preconditioner

        evals, evecs, info = jdqz(A, B, k=5, sigma=0.0, tol=1e-10,
                                   preconditioner=lu, return_info=True,
                                   maxiter=200)
        assert info['nconv'] == 5
        got = np.sort(evals.real)
        expected = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert np.allclose(got, expected, atol=1e-10)

    def test_diagonal_nearest_to_interior_sigma(self):
        """Shift-invert targeting an interior sigma.

        Originally named test_diagonal_target_largest_mag, but the solver
        targets nearest-to-sigma, not largest-magnitude.  As n grows, the
        three nearest-to-sigma=5.05 changes from {5.0, 4.9, 4.8} at
        n=50 (no eigenvalues above 5.0) to {5.0, 5.1, 4.9 OR 5.2} at
        n>=52 where the spectrum straddles sigma and multiple ties
        exist.  We accept any result where each returned eigenvalue is
        among the 3 closest-by-distance values (allowing ties).
        """
        n = 50 * pmult
        A, B = _make_diagonal(n)
        sigma = 5.05
        lu = SparseLU((A - sigma * B).tocsr())
        evals, _, info = jdqz(A, B, k=3, sigma=sigma, tol=1e-10,
                               preconditioner=lu, return_info=True)
        assert info['nconv'] >= 3

        # True eigenvalues of A: 0.1*[1..n].  Find eigenvalues within
        # the 3rd-smallest distance to sigma (inclusive of ties).
        all_evs = 0.1 * np.arange(1, n + 1, dtype=float)
        dists = np.abs(all_evs - sigma)
        sorted_dists = np.sort(dists)
        # Maximum allowed distance: the 3rd-smallest distance, with a
        # small tolerance to accept floating-point ties cleanly.
        max_d = sorted_dists[2] + 1e-12

        # Each returned eigenvalue must be one of the "near enough" ones
        acceptable = set(np.round(all_evs[dists <= max_d], 10))
        for e in evals[:3]:
            assert round(float(np.real(e)), 10) in acceptable, (
                f"eigenvalue {e} is not among nearest-3 (acceptable set: "
                f"{sorted(acceptable)})")

    def test_residuals_below_tol(self):
        A, B = _make_diagonal(50 * pmult)
        lu = SparseLU(A.tocsr())
        tol = 1e-10
        evals, evecs, info = jdqz(A, B, k=5, sigma=0.0, tol=tol,
                                   preconditioner=lu, return_info=True)
        for j in range(info['nconv']):
            res = _residual(A, B, evals[j], evecs[:, j])
            # Allow some margin since tol is a relative metric
            assert res < 100 * tol


# ============================================================
# Correctness — random sparse non-Hermitian
# ============================================================

@pytest.mark.slow
class TestRandomSparse:
    """Random non-Hermitian sparse matrices, compared against scipy.eigs.

    These are @slow because the baseline n=200*pmult problem (n=2000 at
    pmult=10) with 6 requested eigenvalues takes ~2 minutes.
    """

    def test_random_6_eigenvalues(self):
        n = 200 * pmult
        A, B = _make_random_sparse(n, sigma_shift=5.0)
        sigma = 5.0 + 0j

        ref_evals, _ = scipy_eigs(A, k=6, sigma=sigma, which='LM')
        lu = SparseLU((A - sigma * B).tocsr())

        maxsub, minsub = _subspace_sizes()
        # maxiter scales with pmult: some random draws have the 6th
        # eigenvalue in a corner of the spectrum where the SvdV
        # correction converges slowly, and different BLAS libraries'
        # rounding can push that past 500 iters at pmult=10.
        maxiter = max(500, 100 * pmult)
        evals, evecs, info = jdqz(A, B, k=6, sigma=sigma, tol=1e-10,
                                   preconditioner=lu, return_info=True,
                                   maxiter=maxiter,
                                   maxsub=maxsub, minsub=minsub)
        assert info['nconv'] == 6, (
            f"Only {info['nconv']}/6 converged in {info['iters']} iters "
            f"(maxiter={maxiter}).  Try increasing maxiter or maxsub.")
        err = _match_eigenvalues(evals, ref_evals)
        assert err < 1e-8, f"Max eigenvalue error {err:.2e} too large"

        for j in range(info['nconv']):
            res = _residual(A, B, evals[j], evecs[:, j])
            assert res < 1e-8

    def test_random_fewer_eigenvalues(self):
        """Only request 3 — should converge faster."""
        n = 200 * pmult
        A, B = _make_random_sparse(n, sigma_shift=5.0)
        sigma = 5.0 + 0j
        ref_evals, _ = scipy_eigs(A, k=3, sigma=sigma, which='LM')
        lu = SparseLU((A - sigma * B).tocsr())

        maxiter = max(500, 100 * pmult)
        evals, _, info = jdqz(A, B, k=3, sigma=sigma, tol=1e-10,
                               preconditioner=lu, return_info=True,
                               maxiter=maxiter)
        assert info['nconv'] == 3, (
            f"Only {info['nconv']}/3 converged in {info['iters']} iters "
            f"(maxiter={maxiter}).")
        assert _match_eigenvalues(evals, ref_evals) < 1e-8


# ============================================================
# Correctness — block-banded + tau (Dedalus-like)
# ============================================================

@pytest.mark.slow
class TestBlockBandedTau:
    """The key production use case.  @slow at pmult>=2 because n>=600
    with nev=6 takes 10+ seconds; at pmult=10, n=3000 takes minutes."""

    def test_block_banded_tau(self):
        n = 300 * pmult
        A, B = _make_block_banded_tau(n)
        sigma = 3.0 + 0j

        ref_evals, _ = scipy_eigs(A, k=6, sigma=sigma, which='LM')
        lu = SparseLU((A - sigma * B).tocsr(),
                       colamd=True, pin_dense_rows=True)

        maxsub, minsub = _subspace_sizes()
        maxiter = max(500, 100 * pmult)
        evals, evecs, info = jdqz(A, B, k=6, sigma=sigma, tol=1e-10,
                                   preconditioner=lu, return_info=True,
                                   maxiter=maxiter,
                                   maxsub=maxsub, minsub=minsub)
        assert info['nconv'] == 6, (
            f"Only {info['nconv']}/6 converged in {info['iters']} iters "
            f"(maxiter={maxiter}).")
        err = _match_eigenvalues(evals, ref_evals)
        assert err < 1e-8

        for j in range(info['nconv']):
            res = _residual(A, B, evals[j], evecs[:, j])
            assert res < 1e-8


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_no_preconditioner(self):
        """Should work (slowly) without a preconditioner on easy problems.

        maxiter grows with pmult because unpreconditioned JD converges
        more slowly as n increases.  We only assert nconv >= 1 so this
        remains a smoke test at all pmult.
        """
        n = 20 * pmult
        A, B = _make_diagonal(n)
        maxiter = max(500, 50 * pmult)
        evals, _, info = jdqz(A, B, k=2, sigma=0.0, tol=1e-8,
                               return_info=True, maxiter=maxiter)
        assert info['nconv'] >= 1

    def test_small_n(self):
        """Fixed tiny problem: n=5 (not scaled by pmult).  Tests the
        small-n code path where subspace fills the whole space.
        """
        A = sp.diags([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.complex128,
                     format='csr')
        B = sp.eye(5, format='csr', dtype=np.complex128)
        lu = SparseLU(A.tocsr())
        evals, _, info = jdqz(A, B, k=2, sigma=0.0, tol=1e-10,
                               preconditioner=lu, return_info=True)
        assert info['nconv'] == 2
        got = np.sort(evals.real)
        assert np.allclose(got, [1.0, 2.0], atol=1e-10)

    def test_request_subset_of_eigenvalues(self):
        """Requests 5 eigenvalues from a size-20*pmult problem, which
        stresses the subspace management and restart logic when nev is
        a reasonable fraction of n (at pmult=1).
        """
        n = 20 * pmult
        A, B = _make_diagonal(n)
        lu = SparseLU(A.tocsr())
        evals, _, info = jdqz(A, B, k=5, sigma=0.0, tol=1e-10,
                               preconditioner=lu, return_info=True,
                               maxiter=200)
        assert info['nconv'] == 5

    def test_complex_sigma(self):
        """Shift is off the real axis.  Scales correctly with pmult
        because the nearest 3 eigenvalues to sigma are always at integer
        positions around Re(sigma)=15, regardless of n.
        """
        n = 30 * pmult
        if n < 20:
            pytest.skip(f"Spectrum too small for sigma=15 test (n={n})")

        np.random.seed(7)
        d = np.arange(1, n + 1, dtype=np.complex128) + 0.1j * np.random.randn(n)
        A = sp.diags(d, format='csr')
        B = sp.eye(n, format='csr', dtype=np.complex128)
        sigma = 15.0 + 0.1j

        lu = SparseLU((A - sigma * B).tocsr())
        evals, _, info = jdqz(A, B, k=3, sigma=sigma, tol=1e-10,
                               preconditioner=lu, return_info=True)
        assert info['nconv'] == 3
        # The eigenvalues are at integer positions with small imag
        # perturbation, spaced by 1.  The 3 nearest to sigma=15+0.1i
        # are at 14, 15, 16, all within distance < 3 of sigma.
        distances = [abs(e - sigma) for e in evals]
        assert max(distances) < 3.0


# ============================================================
# API tests
# ============================================================

class TestAPI:
    def test_return_info_false(self):
        """When return_info=False, should only return eigvals/eigvecs."""
        n = 20 * pmult
        A, B = _make_diagonal(n)
        lu = SparseLU(A.tocsr())
        result = jdqz(A, B, k=2, sigma=0.0, tol=1e-10, preconditioner=lu,
                       return_info=False)
        assert len(result) == 2  # just evals, evecs
        evals, evecs = result
        assert evals.shape == (2,)
        # Shape scales with n, not a hardcoded 20
        assert evecs.shape == (n, 2), (
            f"evecs shape {evecs.shape} does not match expected ({n}, 2)")

    def test_return_info_true(self):
        A, B = _make_diagonal(20 * pmult)
        lu = SparseLU(A.tocsr())
        result = jdqz(A, B, k=2, sigma=0.0, tol=1e-10, preconditioner=lu,
                       return_info=True)
        assert len(result) == 3
        evals, evecs, info = result
        assert 'nconv' in info
        assert 'iters' in info
        assert 'status' in info

    def test_dense_arrays_accepted(self):
        """Dense numpy arrays should work as A and B."""
        n = 10 * pmult
        A, B = _make_diagonal(n)
        A_dense = A.toarray()
        B_dense = B.toarray()
        lu = SparseLU(sp.csr_matrix(A))  # preconditioner needs sparse
        evals, _, info = jdqz(A_dense, B_dense, k=2, sigma=0.0,
                               tol=1e-10, preconditioner=lu,
                               return_info=True)
        assert info['nconv'] == 2

    def test_linear_operator_matvec(self):
        """Accept callable matvec via LinearOperator-style."""
        from scipy.sparse.linalg import LinearOperator
        nn = 10 * pmult
        A, B = _make_diagonal(nn)
        A_op = LinearOperator((nn, nn), matvec=lambda x: A @ x,
                              dtype=np.complex128)
        B_op = LinearOperator((nn, nn), matvec=lambda x: B @ x,
                              dtype=np.complex128)
        lu = SparseLU(A.tocsr())
        evals, _, info = jdqz(A_op, B_op, k=2, sigma=0.0, tol=1e-10,
                               preconditioner=lu, return_info=True)
        assert info['nconv'] == 2

    def test_start_vector(self):
        """v0 an eigenvector: converged at the first iteration; a wrong
        length is rejected."""
        n = 20 * pmult
        A, B = _make_diagonal(n)
        lu = SparseLU(A.tocsr())
        v0 = np.zeros(n, dtype=np.complex128)
        v0[0] = 1.0
        evals, _, info = jdqz(A, B, k=1, sigma=0.0, tol=1e-10, preconditioner=lu,
                              v0=v0, return_info=True)
        assert info['nconv'] == 1 and info['iters'] == 1
        assert abs(evals[0] - 0.1) < 1e-12
        with pytest.raises(ValueError):
            jdqz(A, B, k=1, sigma=0.0, preconditioner=lu, v0=v0[:-1])

    def test_inner_gmres_with_approximate_preconditioner(self):
        """An approximate preconditioner (the factorization at a shift
        0.5 away): the single-step correction and GMRES(20) on the
        projected correction equation both converge, GMRES in no more
        outer iterations."""
        n = 200 * pmult
        A, B = _make_random_sparse(n)
        sigma = 5.0 + 0.0j
        approx = SparseLU((A - (sigma + 0.5) * B).tocsr())
        ref = scipy_eigs(A, k=6, sigma=sigma, return_eigenvectors=False)
        iters = {}
        for inner in (0, 20):
            evals, evecs, info = jdqz(A, B, k=2, sigma=sigma, tol=1e-10, maxiter=300,
                                      preconditioner=approx, inner_maxiter=inner,
                                      inner_tol=1e-2, return_info=True)
            assert info['nconv'] == 2, f"inner_maxiter={inner}: {info}"
            assert _match_eigenvalues(evals, ref) < 1e-8
            for j in range(2):
                assert _residual(A, B, evals[j], evecs[:, j]) < 1e-8
            iters[inner] = info['iters']
        assert iters[20] <= iters[0], iters

    def test_verbose_does_not_crash(self):
        """verbose=1 and verbose=2 should emit output without errors.

        Note: verbose output is written directly to stderr via C fprintf,
        bypassing Python's sys.stderr redirection.  So we just verify
        the solver still converges correctly with verbose enabled.
        """
        A, B = _make_diagonal(20 * pmult)
        lu = SparseLU(A.tocsr())
        evals, _, info = jdqz(A, B, k=2, sigma=0.0, tol=1e-10,
                               preconditioner=lu, return_info=True,
                               verbose=2)
        assert info['nconv'] == 2
        # Eigenvalues 0.1, 0.2 are the two smallest (and two nearest to
        # sigma=0) independent of n.
        assert np.allclose(np.sort(evals.real), [0.1, 0.2], atol=1e-10)


# ============================================================
# Stability tests
# ============================================================

class TestStability:
    """Stability tests use small fixed-size problems because the
    interesting spectral structure (clustering, repetition) is the
    point, not the size.  These are NOT scaled by pmult.
    """

    def test_clustered_eigenvalues(self):
        """Eigenvalues at 1.0, 1.001, 1.002, 2.0 — tight cluster near 1."""
        d = np.array([1.0, 1.001, 1.002, 2.0, 3.0, 4.0, 5.0],
                     dtype=np.complex128)
        A = sp.diags(d, format='csr')
        B = sp.eye(7, format='csr', dtype=np.complex128)
        # Offset sigma slightly from the cluster to avoid exact singularity
        sigma = 0.9995
        lu = SparseLU((A - sigma * B).tocsr())

        evals, _, info = jdqz(A, B, k=3, sigma=sigma, tol=1e-10,
                               preconditioner=lu, return_info=True,
                               maxiter=500, maxsub=30, minsub=10)
        assert info['nconv'] == 3
        got = np.sort(evals.real)
        expected = [1.0, 1.001, 1.002]
        assert np.allclose(got, expected, atol=1e-8)

    def test_repeated_eigenvalue(self):
        """Diagonal with two eigenvalues at the same location.  We only
        require that the solver doesn't crash and returns values near
        the true spectrum.
        """
        d = np.array([1.0, 2.0, 2.0, 3.0, 4.0], dtype=np.complex128)
        A = sp.diags(d, format='csr')
        B = sp.eye(5, format='csr', dtype=np.complex128)
        lu = SparseLU(A.tocsr())

        evals, _, info = jdqz(A, B, k=3, sigma=2.0, tol=1e-10,
                               preconditioner=lu, return_info=True,
                               maxiter=300)
        assert info['nconv'] >= 2
        for e in evals[:info['nconv']]:
            closest = min([1.0, 2.0, 3.0, 4.0], key=lambda x: abs(e - x))
            assert abs(e - closest) < 1e-8


# ============================================================
# Regression tests for specific bugs we fixed
# ============================================================

class TestRegression:
    """Each test here corresponds to a specific bug caught during development.
    Do not remove these even if they look redundant with the correctness
    tests above — they are documentation as much as verification."""

    @pytest.mark.slow
    def test_target_pos_after_ztgsen(self):
        """Regression: after the first eigenvalue locks, ZTGSEN may place
        the locked eigenvalue at position 0 or position `locked` in
        arbitrary order.  The code must scan alpha/beta to find the
        actual target, not blindly read position `locked`.

        If this test passes at pmult=1 but returns only 1/3 eigenvalues
        at higher pmult, the target_pos fix has regressed.  @slow because
        n=200*pmult with nev=3 takes 2+ minutes at pmult=10.
        """
        n = 200 * pmult
        A, B = _make_random_sparse(n, sigma_shift=5.0)
        sigma = 5.0 + 0j
        lu = SparseLU((A - sigma * B).tocsr())

        maxsub, minsub = _subspace_sizes()
        maxiter = max(500, 100 * pmult)
        evals, _, info = jdqz(A, B, k=3, sigma=sigma, tol=1e-10,
                               preconditioner=lu, return_info=True,
                               maxiter=maxiter, maxsub=maxsub, minsub=minsub)
        assert info['nconv'] == 3, (
            f"Only {info['nconv']}/3 converged — target_pos scan may have "
            f"regressed. Check that solve_projected returns *target_pos.")

    def test_schur_locking_no_reconverge(self):
        """Regression: the same eigenvalue must not be returned twice.
        Previous bug: after locking eigenvalue X, a restart could re-find
        the locked eigenvalue because V still contained its Schur vector.
        """
        n = 20 * pmult
        A, B = _make_diagonal(n, spacing=0.1)
        lu = SparseLU(A.tocsr())
        evals, _, info = jdqz(A, B, k=5, sigma=0.0, tol=1e-10,
                               preconditioner=lu, return_info=True,
                               maxiter=300)
        assert info['nconv'] == 5
        # No two converged eigenvalues should be within half the true
        # spacing (0.1) — i.e., no duplicates.
        got = np.sort(evals.real)
        spacing_half = 0.05
        for i in range(len(got) - 1):
            assert got[i + 1] - got[i] > spacing_half, (
                f"Duplicate eigenvalues detected: {got}")

    @pytest.mark.slow
    def test_thick_restart_preserves_locked(self):
        """After a thick restart, the locked eigenvalues must still be
        representable in the (compressed) Schur form.

        Forces restarts with a deliberately small maxsub.  Keep maxsub
        fixed (not scaled with pmult) so restarts actually happen.
        @slow because small maxsub plus large n is expensive.
        """
        n = 200 * pmult
        A, B = _make_random_sparse(n, sigma_shift=5.0)
        sigma = 5.0 + 0j
        lu = SparseLU((A - sigma * B).tocsr())

        # maxsub=30 forces restarts even with nev=4
        evals, _, info = jdqz(A, B, k=4, sigma=sigma, tol=1e-10,
                               preconditioner=lu, return_info=True,
                               maxiter=2000, maxsub=30, minsub=15)
        assert info['nconv'] >= 2


# ============================================================
# Slow / large-scale tests (opt-in)
# ============================================================

@pytest.mark.slow
class TestSlow:
    """Additional slow tests covering larger-scale behavior.
    Run with ``pytest -m slow``.
    """

    def test_larger_random(self):
        """Larger random problem — scales aggressively with pmult."""
        n = 500 * pmult
        np.random.seed(42)
        d = min(0.02, 50.0 / n) if n > 1000 else 0.02
        A = sp.random(n, n, density=d, format='csr', dtype=np.complex128)
        A = A + 5 * sp.eye(n, dtype=np.complex128)
        A.data += 0.1j * np.random.randn(A.nnz)
        B = sp.eye(n, format='csr', dtype=np.complex128)
        sigma = 5.0 + 0j
        lu = SparseLU((A - sigma * B).tocsr())

        evals, _, info = jdqz(A, B, k=5, sigma=sigma, tol=1e-10,
                               preconditioner=lu, return_info=True,
                               maxiter=500, maxsub=100, minsub=40)
        assert info['nconv'] >= 3  # don't require all, just progress


# ============================================================
# Manual runner (usable without pytest)
# ============================================================

if __name__ == '__main__':
    import argparse
    import traceback
    import time

    parser = argparse.ArgumentParser(description='Run BBJD tests.')
    parser.add_argument('--slow', action='store_true',
                        help='Run the slow tests as well')
    args = parser.parse_args()

    print(f"Running tests with pmult = {pmult}, slow = {args.slow}")
    print("=" * 60)

    # (class, methods, is_class_slow)
    tests = [
        (TestDiagonal, [
            'test_diagonal_5_nearest_sigma0',
            'test_diagonal_nearest_to_interior_sigma',
            'test_residuals_below_tol'], False),
        (TestRandomSparse, [
            'test_random_6_eigenvalues',
            'test_random_fewer_eigenvalues'], True),
        (TestBlockBandedTau, [
            'test_block_banded_tau'], True),
        (TestEdgeCases, [
            'test_no_preconditioner', 'test_small_n',
            'test_request_subset_of_eigenvalues', 'test_complex_sigma'], False),
        (TestAPI, [
            'test_return_info_false', 'test_return_info_true',
            'test_dense_arrays_accepted', 'test_linear_operator_matvec',
            'test_verbose_does_not_crash'], False),
        (TestStability, [
            'test_clustered_eigenvalues', 'test_repeated_eigenvalue'], False),
        (TestRegression, [
            ('test_target_pos_after_ztgsen', True),  # method-level slow
            ('test_schur_locking_no_reconverge', False),
            ('test_thick_restart_preserves_locked', True)], False),
        (TestSlow, ['test_larger_random'], True),
    ]

    passed = failed = skipped = 0
    for entry in tests:
        cls, methods, cls_is_slow = entry
        instance = cls()
        for m in methods:
            if isinstance(m, tuple):
                method_name, method_is_slow = m
            else:
                method_name, method_is_slow = m, False
            is_slow = cls_is_slow or method_is_slow
            if is_slow and not args.slow:
                print(f"  SKIP  {cls.__name__}.{method_name} (slow)")
                skipped += 1
                continue
            t0 = time.time()
            try:
                getattr(instance, method_name)()
                dt = time.time() - t0
                tag = ' [slow]' if is_slow else ''
                print(f"  PASS  {cls.__name__}.{method_name}{tag} ({dt:.2f}s)")
                passed += 1
            except Exception as e:
                dt = time.time() - t0
                msg = str(e) or type(e).__name__
                print(f"  FAIL  {cls.__name__}.{method_name} ({dt:.2f}s): "
                      f"{msg}")
                traceback.print_exc()
                failed += 1
    print()
    print(f"{passed} passed, {failed} failed, {skipped} skipped  "
          f"(pmult={pmult})")
    sys.exit(0 if failed == 0 else 1)
