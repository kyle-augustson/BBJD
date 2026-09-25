"""
bbjd.py — Block-Bordered Jacobi-Davidson eigensolver
=====================================================

Python interface to the JDQZ C core.  Provides a single user-facing
function ``jdqz()`` with an API modelled on scipy.sparse.linalg.eigs.

Usage
-----
    from spdee.solvers.bbjd import jdqz

    eigenvalues, eigenvectors = jdqz(
        A, B,
        k=10,
        sigma=0.0+0j,
        tol=1e-10,
    )

For best performance with block-banded-plus-border matrices from
spectral PDE discretizations, supply a preconditioner via the
``SparseLU`` class of ``spdee.solvers.fplus``::

    from spdee.solvers.fplus import SparseLU

    eigenvalues, eigenvectors = jdqz(
        A, B,
        k=10,
        sigma=0.0+0j,
        preconditioner=SparseLU(A - sigma * B,
                                colamd=True,
                                pin_dense_rows=True),
    )
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

try:
    from . import _bbjd_c            # as spdee.solvers.bbjd
except ImportError:                  # standalone use from this directory
    try:
        import _bbjd_c
    except ImportError:              # extension not built: jdqz raises on use
        _bbjd_c = None

# Target enum values (must match bbjd_target in bbjd.h)
_TARGET_MAP = {
    'nearest':       0,
    'LM':            1,  # largest magnitude
    'SM':            2,  # smallest magnitude
    'LR':            3,  # largest real
    'SR':            4,  # smallest real
    'LI':            5,  # largest imaginary
    'SI':            6,  # smallest imaginary
}


def jdqz(
    A,
    B=None,
    k: int = 6,
    sigma: complex = 0.0,
    which: str = 'nearest',
    tol: float = 1e-10,
    maxiter: int = 200,
    maxsub: int = 0,
    minsub: int = 0,
    inner_maxiter: int = 0,
    inner_tol: float = 1e-2,
    preconditioner=None,
    v0=None,
    verbose: int = 0,
    return_info: bool = False,
):
    """
    Jacobi-Davidson QZ eigensolver for generalized eigenproblems.

    Solves  A @ x = λ * B @ x  for a few eigenvalues nearest to sigma.

    Parameters
    ----------
    A : sparse matrix or LinearOperator, shape (n, n)
        The left-hand-side operator.

    B : sparse matrix or LinearOperator, shape (n, n), optional
        The right-hand-side (mass) operator.  Default: identity.

    k : int
        Number of eigenvalues to compute.

    sigma : complex
        Target shift.  Eigenvalues nearest to sigma are sought first.

    which : str
        Eigenvalue selection criterion:
        'nearest' (default), 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'.

    tol : float
        Convergence tolerance for eigenvalue residuals.

    maxiter : int
        Maximum number of outer JD iterations.

    maxsub : int
        Maximum search subspace dimension before restart.
        Default 0 = auto (min(2*k+10, n)).

    minsub : int
        Subspace dimension after thick restart.
        Default 0 = auto (k+2).

    inner_maxiter : int
        0 (default): the correction equation is solved by one application
        of the preconditioner, the Sleijpen-van der Vorst step
        t = z - s (u^H z)/(u^H s) with z = P^-1 r, s = P^-1 u, which is
        the exact correction when the preconditioner is the exact
        factorization of A - sigma B.  Otherwise the maximum number of
        GMRES iterations on the projected correction equation, left
        preconditioned, for an approximate preconditioner.

    inner_tol : float
        Relative tolerance for the inner GMRES solve.

    preconditioner : object with .solve(b) method, optional
        Preconditioner for the correction equation, approximating
        (A - sigma*B)^{-1}.  Typically a SparseLU factorization.
        If None, no preconditioning is used.

    v0 : ndarray, shape (n,), optional
        Starting vector of the search space (default: random).

    verbose : int
        0 = silent, 1 = convergence summary, 2 = per-iteration info.

    return_info : bool
        If True, return a dict with convergence info.

    Returns
    -------
    eigenvalues : ndarray, shape (nconv,)
        Converged eigenvalues (complex128).

    eigenvectors : ndarray, shape (n, nconv)
        Corresponding eigenvectors (complex128), one per column.

    info : dict (only if return_info=True)
        'nconv': number of converged eigenvalues
        'iters': outer iterations used
        'status': 0=success, 1=max_iter, negative=error
        'residuals': residual norms for each eigenvalue

    Raises
    ------
    ValueError
        If inputs are invalid.
    RuntimeError
        If the solver encounters an unrecoverable error.
    """
    # ---- Validate inputs ----
    if isinstance(A, np.ndarray):
        A = sp.csr_matrix(A)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"A must be square; got shape {A.shape}")

    if B is None:
        B_op = sp.eye(n, dtype=np.complex128, format='csr')
    elif isinstance(B, np.ndarray):
        B_op = sp.csr_matrix(B)
    else:
        B_op = B

    if which not in _TARGET_MAP:
        raise ValueError(
            f"which must be one of {list(_TARGET_MAP.keys())}; got {which!r}")

    sigma = complex(sigma)
    target = _TARGET_MAP[which]

    # ---- Build matvec callables ----
    # Convert to scipy sparse for fast matvec if possible
    if sp.issparse(A):
        A_csc = A.astype(np.complex128).tocsc()
        def matvec_A(x):
            return A_csc @ x
    elif hasattr(A, 'matvec'):
        def matvec_A(x):
            return np.asarray(A.matvec(x), dtype=np.complex128)
    elif callable(A):
        matvec_A = A
    else:
        raise ValueError("A must be a sparse matrix, LinearOperator, or callable")

    if sp.issparse(B_op):
        B_csc = B_op.astype(np.complex128).tocsc()
        def matvec_B(x):
            return B_csc @ x
    elif hasattr(B_op, 'matvec'):
        def matvec_B(x):
            return np.asarray(B_op.matvec(x), dtype=np.complex128)
    elif callable(B_op):
        matvec_B = B_op
    else:
        raise ValueError("B must be a sparse matrix, LinearOperator, or callable")

    # ---- Build preconditioner callable ----
    if preconditioner is not None:
        if hasattr(preconditioner, 'solve'):
            pc = preconditioner
            def precond_solve(x):
                return np.asarray(pc.solve(x), dtype=np.complex128)
        elif callable(preconditioner):
            precond_solve = preconditioner
        else:
            raise ValueError(
                "preconditioner must have a .solve(b) method or be callable")
    else:
        precond_solve = None

    if v0 is not None:
        v0 = np.ascontiguousarray(v0, dtype=np.complex128)
        if v0.shape != (n,):
            raise ValueError(f"v0 must have shape ({n},); got {v0.shape}")

    # ---- Call C solver ----
    if _bbjd_c is None:
        raise ImportError("spdee.solvers.bbjd: the C extension _bbjd_c is not built; "
                          "run spdee/solvers/build_ext.sh")
    evals, evecs, resid, nconv, iters, status = _bbjd_c.jdqz_c(
        n=n,
        nev=k,
        sigma=sigma,
        target=target,
        matvec_A=matvec_A,
        matvec_B=matvec_B,
        precond_solve=precond_solve,
        tol=tol,
        maxiter=maxiter,
        maxsub=maxsub,
        minsub=minsub,
        inner_maxiter=inner_maxiter,
        inner_tol=inner_tol,
        verbose=verbose,
        v0=v0,
    )

    if status < 0:
        status_msg = {
            -1: "memory allocation failed",
            -2: "invalid parameters",
            -3: "callback error (matvec or preconditioner failed)",
            -4: "LAPACK error in projected eigenproblem",
            -5: "subspace breakdown",
        }
        msg = status_msg.get(status, f"unknown error (status={status})")
        raise RuntimeError(f"JDQZ solver failed: {msg}")

    if status == 1 and verbose >= 1:
        import warnings
        warnings.warn(
            f"JDQZ: only {nconv}/{k} eigenvalues converged "
            f"in {iters} iterations.",
            RuntimeWarning,
            stacklevel=2,
        )

    if return_info:
        info = {
            'nconv': nconv,
            'iters': iters,
            'status': status,
            'residuals': resid,
        }
        return evals, evecs, info
    else:
        return evals, evecs
