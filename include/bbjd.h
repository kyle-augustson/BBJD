/*
 * bbjd.h — Block-Bordered Jacobi-Davidson Eigensolver
 *
 * Solves the generalized non-Hermitian eigenproblem  A x = λ B x
 * using the JDQZ variant (Fokkema, Sleijpen, van der Vorst 1999)
 * with preconditioned GMRES for the correction equation.
 *
 * Designed for pencils arising from spectral PDE discretizations:
 * block-banded interior with dense boundary-condition (tau) rows.
 *
 * All dense subspace vectors are stored column-major (Fortran order)
 * for BLAS/LAPACK compatibility.  Complex arithmetic throughout.
 *
 * ============================================================
 *  DEPLOYMENT MODEL (typical PDE eigenspectrum sweep)
 * ============================================================
 *  Outer:  MPI ranks, one per guess eigenvalue (embarrassingly parallel).
 *          Each rank independently factors (A - σ B) and runs BBJD.
 *
 *  Inner:  Threads shared by the LU factorization/solve and BBJD's
 *          BLAS calls.  These never run concurrently — LU solve is
 *          called sequentially from within each BBJD iteration — so
 *          there is no oversubscription risk if every threaded
 *          library (OpenBLAS, SparseLU, BBJD's OpenMP first-touch)
 *          honors the same OMP_NUM_THREADS setting.
 *
 *  Environment variables expected at launch:
 *    OMP_NUM_THREADS    = <threads per rank>  (e.g., 32 for 4x32 layout)
 *    OMP_PROC_BIND      = close
 *    OMP_PLACES         = cores
 *    OPENBLAS_NUM_THREADS = $OMP_NUM_THREADS
 *
 *  Pin MPI ranks to NUMA domains:
 *    mpirun -np $NRANKS \
 *      numactl --cpunodebind=$(( RANK / ranks_per_numa )) \
 *              --membind=$(( RANK / ranks_per_numa )) \
 *      python eigensweep.py
 *
 *  BBJD's workspace allocator uses first-touch initialization so all
 *  large arrays (V, W, AV, BV ~ n × maxsub) are placed on the
 *  rank's local NUMA node automatically.
 *
 *  MEMORY at n=10^6, nev=64, maxsub=200:
 *    V, W, AV, BV, swap_nm:  ~3 GB each × 5 = 15 GB
 *    Q_right, Q_left:         ~1 GB combined
 *    gmres_V (inner_maxiter=50): ~0.8 GB
 *    Total per rank:           ~17 GB + matvec/LU memory
 * ============================================================
 */
#ifndef BBJD_H
#define BBJD_H

#include <complex.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Basic types
 * ================================================================ */
typedef double complex  bbjd_zomplex;
typedef int64_t         bbjd_int;

/* ================================================================
 * Return codes
 * ================================================================ */
typedef enum {
    BBJD_SUCCESS           =  0,
    BBJD_MAX_ITER          =  1,  /* converged fewer than requested  */
    BBJD_ERR_ALLOC         = -1,
    BBJD_ERR_INVALID       = -2,
    BBJD_ERR_CALLBACK      = -3,
    BBJD_ERR_LAPACK        = -4,
    BBJD_ERR_BREAKDOWN     = -5,
} bbjd_status;

/* ================================================================
 * Eigenvalue target selection
 * ================================================================ */
typedef enum {
    BBJD_TARGET_NEAREST,        /* nearest to sigma              */
    BBJD_TARGET_LARGEST_MAG,    /* largest  |λ|                  */
    BBJD_TARGET_SMALLEST_MAG,   /* smallest |λ|                  */
    BBJD_TARGET_LARGEST_REAL,   /* largest  Re(λ)                */
    BBJD_TARGET_SMALLEST_REAL,  /* smallest Re(λ)                */
    BBJD_TARGET_LARGEST_IMAG,   /* largest  Im(λ)                */
    BBJD_TARGET_SMALLEST_IMAG,  /* smallest Im(λ)                */
} bbjd_target;

/* ================================================================
 * Callback function types
 *
 * matvec:  y = Op @ x              (length n vectors)
 * precond: z = M^{-1} r            (preconditioner solve)
 *
 * Return 0 on success, nonzero on error.
 * ================================================================ */
typedef int (*bbjd_matvec_fn)(bbjd_int n,
                              const bbjd_zomplex *x,
                              bbjd_zomplex *y,
                              void *data);

typedef int (*bbjd_precond_fn)(bbjd_int n,
                               const bbjd_zomplex *x,
                               bbjd_zomplex *y,
                               void *data);

/* ================================================================
 * Parameters
 * ================================================================ */
typedef struct {
    /* Problem size */
    bbjd_int        n;

    /* What to compute */
    bbjd_int        nev;        /* number of eigenvalues wanted     */
    bbjd_zomplex    sigma;      /* target shift                     */
    bbjd_target     target;     /* selection strategy                */

    /* Subspace dimensions */
    bbjd_int        maxsub;     /* max search subspace dim          */
    bbjd_int        minsub;     /* subspace dim after restart       */

    /* Convergence */
    double          tol;        /* outer convergence tolerance      */
    bbjd_int        maxiter;    /* max outer JD iterations          */

    /* Inner (correction equation) solver: inner_maxiter = 0 takes the
     * single preconditioned Sleijpen-van der Vorst step (exact when the
     * preconditioner is the factorization of A - sigma B); > 0 solves the
     * projected correction equation by left-preconditioned GMRES with at
     * most that many iterations to the relative tolerance inner_tol.   */
    bbjd_int        inner_maxiter;
    double          inner_tol;

    /* Starting vector of the search space (length n); NULL => random */
    const bbjd_zomplex *v0;

    /* Operators */
    bbjd_matvec_fn  matvec_A;
    bbjd_matvec_fn  matvec_B;
    bbjd_precond_fn precond;    /* NULL => no preconditioning       */
    void           *op_data;    /* opaque data for all callbacks    */

    /* Verbosity: 0=silent, 1=summary, 2=per-iteration              */
    int             verbose;
} bbjd_params;

/* ================================================================
 * Result
 * ================================================================ */
typedef struct {
    bbjd_int        nconv;      /* number of converged eigenvalues  */
    bbjd_zomplex   *eigenvalues;  /* length nconv                   */
    bbjd_zomplex   *eigenvectors; /* n × nconv, column-major        */
    double         *residuals;    /* length nconv                   */
    bbjd_int        iters;        /* outer iterations used          */
    bbjd_status     status;
} bbjd_result;

/* ================================================================
 * Workspace  (opaque, allocated by bbjd_workspace_alloc)
 *
 * All memory for the iteration is held here so the solver
 * does zero dynamic allocation after initialisation.
 * ================================================================ */
typedef struct {
    bbjd_int    n, maxsub, nev;

    /* Search and test subspaces: V, W  (n × maxsub, col-major) */
    bbjd_zomplex *V;            /* search space                     */
    bbjd_zomplex *W;            /* W = A V  (test via B-inner prod) */
    bbjd_zomplex *AV;           /* A @ V columns                    */
    bbjd_zomplex *BV;           /* B @ V columns                    */

    /* Projected pencil: (MA, MB) = (W^H A V, W^H B V)
     * Stored maxsub × maxsub, column-major                        */
    bbjd_zomplex *MA;
    bbjd_zomplex *MB;

    /* Schur decomposition of projected pencil:
     * MA = ZA * SA * ZA^H,  MB = ZB * SB * ZB^H
     * Actually generalized Schur: (SA, SB) upper triangular       */
    bbjd_zomplex *SA, *SB;      /* maxsub × maxsub                 */
    bbjd_zomplex *ZA, *ZB;      /* maxsub × maxsub                 */

    /* Converged Schur vectors locked in place */
    bbjd_int     nlock;         /* number of locked eigenvalues     */

    /* Current subspace dimension */
    bbjd_int     kdim;

    /* GMRES workspace */
    bbjd_zomplex *gmres_V;      /* n × (inner_maxiter+1)           */
    bbjd_zomplex *gmres_H;      /* (inner_maxiter+1) × inner_maxiter */
    bbjd_zomplex *gmres_cs;     /* Givens cosines,  inner_maxiter  */
    bbjd_zomplex *gmres_sn;     /* Givens sines,    inner_maxiter  */
    bbjd_zomplex *gmres_g;      /* residual vector, inner_maxiter+1*/
    bbjd_int      inner_maxiter;

    /* Scratch vectors (length n each) */
    bbjd_zomplex *tmp1, *tmp2, *tmp3, *tmp4;

    /* Restart swap buffer: n × maxsub, used for thick-restart
     * zgemm(V, ZB) → swap → memcpy into V, avoiding per-iter malloc.
     * Also used for zgemm(AV, ZB) and zgemm(BV, ZB) sequentially. */
    bbjd_zomplex *swap_nm;

    /* Deflation projectors of the correction equation (both n × (nev+1)).
     *   Q_right[:, 0..nlock-1] = orthonormalized converged eigenvectors
     *   Q_right[:, nlock]      = current Ritz vector u, orthogonalized
     *   Q_left[:, j]           = orthonormalized test-space images
     *                            nu0 A q_j + mu0 B q_j of the columns  */
    bbjd_zomplex *Q_right;
    bbjd_zomplex *Q_left;

    /* Persistent workspace for block orthogonalization coefficients
     * (length maxsub+nev to cover all use cases)                    */
    bbjd_zomplex *h_work;

} bbjd_workspace;

/* ================================================================
 * Public API
 * ================================================================ */

/* Set params to sensible defaults.  Caller must still set
 * n, nev, matvec_A, matvec_B, and (optionally) precond.          */
void bbjd_params_init(bbjd_params *p);

/* Allocate / free workspace */
bbjd_workspace *bbjd_workspace_alloc(const bbjd_params *p);
void            bbjd_workspace_free(bbjd_workspace *ws);

/* Main solver entry point */
bbjd_status bbjd_solve(const bbjd_params *params,
                        bbjd_workspace    *ws,
                        bbjd_result       *result);

/* Free result arrays (eigenvalues, eigenvectors, residuals) */
void bbjd_result_free(bbjd_result *result);

/* ================================================================
 * Internal BLAS/LAPACK helpers  (used by the solver, exposed
 * for testing)
 * ================================================================ */

/* Modified Gram-Schmidt orthogonalization of v against columns
 * of Q (n × k, column-major).  v is modified in place.
 * Returns the norm of v after orthogonalization.                  */
double bbjd_mgs_orthogonalize(bbjd_int n, bbjd_int k,
                               const bbjd_zomplex *Q,
                               bbjd_zomplex *v);

/* GMRES for the projected correction equation:
 *   (I - q q^H)(A - theta B)(I - p p^H) z = -r
 * with preconditioner M.
 *
 * q, p:     projection vectors (length n)
 * theta:    current Ritz value
 * r:        residual (length n, input)
 * z:        correction (length n, output)
 *
 * Returns 0 on success.                                           */
int bbjd_correction_gmres(
    bbjd_int n,
    const bbjd_zomplex *q,       /* left  projection vector        */
    const bbjd_zomplex *p,       /* right projection vector        */
    bbjd_zomplex        theta_a, /* Ritz numerator   (alpha)       */
    bbjd_zomplex        theta_b, /* Ritz denominator (beta)        */
    const bbjd_zomplex *r,       /* residual to solve              */
    bbjd_zomplex       *z,       /* output correction vector       */
    bbjd_matvec_fn      matvec_A,
    bbjd_matvec_fn      matvec_B,
    bbjd_precond_fn     precond,
    void               *op_data,
    bbjd_int            maxiter,
    double              tol,
    bbjd_workspace     *ws);

#ifdef __cplusplus
}
#endif

#endif /* BBJD_H */
