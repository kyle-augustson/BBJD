/*
 * jd_core.c — Jacobi-Davidson QZ (JDQZ) eigensolver core
 *
 * References:
 *   [1] Fokkema, Sleijpen, van der Vorst. SIAM J. Sci. Comput. 20(1), 1998.
 *   [2] Sleijpen, van der Vorst. SIAM Review 42(2), 2000.
 *
 * Memory layout: all dense subspace matrices are column-major.
 */

#include "bbjd.h"
#include "bbjd_blas.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <float.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static const bbjd_zomplex ZERO = 0.0;
static const bbjd_zomplex ONE  = 1.0;

static inline bbjd_zomplex *col(bbjd_zomplex *M, bbjd_int n, bbjd_int j)
{ return M + j * n; }
static inline const bbjd_zomplex *colc(const bbjd_zomplex *M, bbjd_int n, bbjd_int j)
{ return M + j * n; }

/* ================================================================
 * NUMA-aware aligned allocation with first-touch initialization.
 *
 * Strategy: allocate cache-aligned memory (64 B on AMD Epyc), then
 * use an OpenMP parallel loop to first-touch each page in parallel.
 * The Linux kernel's default NUMA policy (MPOL_DEFAULT = first-touch)
 * places each page on the node of the thread that first accesses it.
 *
 * For this to be effective, the caller must run the solver with the
 * same thread-to-core binding that was used for this allocation.
 * Tools: OMP_PROC_BIND=close OMP_PLACES=cores (for compact layout)
 * or OMP_PROC_BIND=spread for wide layout.
 *
 * For MPI ranks, use `numactl --membind` or `--cpunodebind` to
 * restrict each rank to a single NUMA node before allocation.
 * ================================================================ */
static void *bbjd_aligned_alloc_ft(size_t nbytes)
{
    const size_t align = 64;
    /* Round up to multiple of align for aligned_alloc requirement */
    size_t nb = (nbytes + align - 1) & ~(align - 1);
    void *p = aligned_alloc(align, nb);
    if (!p) return NULL;

    /* First-touch in parallel: touching one byte per 4KB page
     * suffices to bind that page to the touching thread's NUMA node. */
    char *cp = (char *)p;
    size_t npages = (nb + 4095) / 4096;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < npages; i++) {
        cp[i * 4096] = 0;
    }
#else
    /* Without OpenMP: single-threaded zero */
    memset(cp, 0, nb);
#endif
    return p;
}

/* ================================================================ */
void bbjd_params_init(bbjd_params *p)
{
    memset(p, 0, sizeof(*p));
    p->nev           = 6;
    p->sigma         = 0.0;
    p->target        = BBJD_TARGET_NEAREST;
    p->tol           = 1e-10;
    p->maxiter       = 200;
    p->inner_maxiter = 0;
    p->inner_tol     = 1e-2;
    p->v0            = NULL;
}

/* ================================================================ */
bbjd_workspace *bbjd_workspace_alloc(const bbjd_params *p)
{
    bbjd_int n      = p->n;
    bbjd_int nev    = p->nev;
    /* Default maxsub: 3*nev+10 (minimum 40)
     * For n=10^6, nev=64 -> maxsub=202. V+AV+BV+W = 4 * 202 * n * 16 B
     *                                             = 51 GB.  User should
     * pass a smaller maxsub if memory is tight.                     */
    bbjd_int maxsub_auto = 3*nev + 10;
    if (maxsub_auto < 40) maxsub_auto = 40;
    bbjd_int maxsub = p->maxsub > 0 ? p->maxsub
                    : (maxsub_auto < n ? maxsub_auto : n);
    bbjd_int ims    = p->inner_maxiter > 0 ? p->inner_maxiter : 1;

    bbjd_workspace *ws = (bbjd_workspace *)calloc(1, sizeof(*ws));
    if (!ws) return NULL;

    ws->n = n;  ws->maxsub = maxsub;  ws->nev = nev;
    ws->nlock = 0;  ws->kdim = 0;  ws->inner_maxiter = ims;

    size_t ns = (size_t)n * maxsub;
    size_t ss = (size_t)maxsub * maxsub;

    /* Large n×maxsub arrays get NUMA-aware first-touch allocation.
     * At n=10^6 these are ~3 GB each; placing them correctly across
     * NUMA domains is critical for memory bandwidth.                  */
    ws->V  = bbjd_aligned_alloc_ft(ns * sizeof(bbjd_zomplex));
    ws->W  = bbjd_aligned_alloc_ft(ns * sizeof(bbjd_zomplex));
    ws->AV = bbjd_aligned_alloc_ft(ns * sizeof(bbjd_zomplex));
    ws->BV = bbjd_aligned_alloc_ft(ns * sizeof(bbjd_zomplex));
    ws->swap_nm = bbjd_aligned_alloc_ft(ns * sizeof(bbjd_zomplex));

    /* Small maxsub×maxsub projected matrices — calloc is fine */
    ws->MA = calloc(ss, sizeof(bbjd_zomplex));
    ws->MB = calloc(ss, sizeof(bbjd_zomplex));
    ws->SA = calloc(ss, sizeof(bbjd_zomplex));
    ws->SB = calloc(ss, sizeof(bbjd_zomplex));
    ws->ZA = calloc(ss, sizeof(bbjd_zomplex));
    ws->ZB = calloc(ss, sizeof(bbjd_zomplex));

    /* GMRES workspace: n×(inner_maxiter+1) can be large too */
    ws->gmres_V  = bbjd_aligned_alloc_ft((size_t)n * (ims + 1)
                                          * sizeof(bbjd_zomplex));
    ws->gmres_H  = calloc((size_t)(ims + 1) * ims, sizeof(bbjd_zomplex));
    ws->gmres_cs = calloc(ims, sizeof(bbjd_zomplex));
    ws->gmres_sn = calloc(ims, sizeof(bbjd_zomplex));
    ws->gmres_g  = calloc(ims + 1, sizeof(bbjd_zomplex));

    /* Scratch n-vectors: first-touch helps these too */
    ws->tmp1 = bbjd_aligned_alloc_ft(n * sizeof(bbjd_zomplex));
    ws->tmp2 = bbjd_aligned_alloc_ft(n * sizeof(bbjd_zomplex));
    ws->tmp3 = bbjd_aligned_alloc_ft(n * sizeof(bbjd_zomplex));
    ws->tmp4 = bbjd_aligned_alloc_ft(n * sizeof(bbjd_zomplex));

    /* Deflation projectors: n×(nev+1), ~520 MB at n=10^6 nev=64 */
    ws->Q_right = bbjd_aligned_alloc_ft((size_t)n * (nev + 1)
                                         * sizeof(bbjd_zomplex));
    ws->Q_left  = bbjd_aligned_alloc_ft((size_t)n * (nev + 1)
                                         * sizeof(bbjd_zomplex));

    /* Persistent h_work: block orthogonalization and the Ritz-vector
     * back-substitution (2 maxsub + nev + 4 entries) */
    ws->h_work = calloc((size_t)(2 * maxsub + nev + 4), sizeof(bbjd_zomplex));

    if (!ws->V || !ws->W || !ws->AV || !ws->BV ||
        !ws->MA || !ws->MB || !ws->SA || !ws->SB || !ws->ZA || !ws->ZB ||
        !ws->gmres_V || !ws->gmres_H || !ws->gmres_cs ||
        !ws->gmres_sn || !ws->gmres_g ||
        !ws->tmp1 || !ws->tmp2 || !ws->tmp3 || !ws->tmp4 ||
        !ws->swap_nm || !ws->Q_right || !ws->Q_left || !ws->h_work) {
        bbjd_workspace_free(ws);
        return NULL;
    }
    return ws;
}

void bbjd_workspace_free(bbjd_workspace *ws)
{
    if (!ws) return;
    free(ws->V);  free(ws->W);  free(ws->AV); free(ws->BV);
    free(ws->MA); free(ws->MB);
    free(ws->SA); free(ws->SB); free(ws->ZA); free(ws->ZB);
    free(ws->gmres_V); free(ws->gmres_H);
    free(ws->gmres_cs); free(ws->gmres_sn); free(ws->gmres_g);
    free(ws->tmp1); free(ws->tmp2); free(ws->tmp3); free(ws->tmp4);
    free(ws->swap_nm);
    free(ws->Q_right); free(ws->Q_left);
    free(ws->h_work);
    free(ws);
}

void bbjd_result_free(bbjd_result *r)
{
    if (!r) return;
    free(r->eigenvalues);  free(r->eigenvectors);  free(r->residuals);
    r->eigenvalues = NULL; r->eigenvectors = NULL;  r->residuals = NULL;
}

/* ================================================================
 * MGS with DGKS reorthogonalization (two passes)
 * ================================================================ */
/* ================================================================
 * Block-orthogonalize v against Q[:,0:k] using BLAS-2 (zgemv).
 *
 * This is classical Gram-Schmidt with DGKS reorthogonalization:
 *   h = Q^H v                    (one zgemv, "C"onjugate-transpose)
 *   v = v - Q h                  (one zgemv, "N"o-transpose)
 *
 * For large k (e.g., k=200 columns at n=10^6), this is 2-3x faster
 * than MGS because:
 *   1. BLAS-2 is cache-blocked inside OpenBLAS
 *   2. Fewer kernel launches (2 vs 2k)
 *   3. Q is traversed in one contiguous sweep per pass, improving
 *      hardware prefetcher effectiveness
 *
 * CGS2 with DGKS: if ||v_after|| < 1/sqrt(2) * ||v_before||,
 * do a second pass.  Guaranteed orthogonality to O(eps*kappa(Q)).
 *
 * Workspace: caller provides h_work of size >= k.
 * ================================================================ */
double bbjd_block_orthogonalize(bbjd_int n, bbjd_int k,
                                 const bbjd_zomplex *Q, bbjd_zomplex *v,
                                 bbjd_zomplex *h_work)
{
    if (k == 0) {
        blas_int bn = (blas_int)n, one = 1;
        return dznrm2_(&bn, v, &one);
    }

    blas_int bn = (blas_int)n, bk = (blas_int)k, one = 1;
    bbjd_zomplex alpha_one =  1.0;
    bbjd_zomplex neg_one   = -1.0;
    bbjd_zomplex zero      =  0.0;

    double v_norm_before = dznrm2_(&bn, v, &one);

    /* Pass 1: h = Q^H v, then v = v - Q h */
    zgemv_("C", &bn, &bk, &alpha_one, Q, &bn, v, &one,
           &zero, h_work, &one);
    zgemv_("N", &bn, &bk, &neg_one, Q, &bn, h_work, &one,
           &alpha_one, v, &one);

    double v_norm_after = dznrm2_(&bn, v, &one);

    /* DGKS: reorthogonalize if significant cancellation */
    if (v_norm_after < 0.7071 * v_norm_before) {
        zgemv_("C", &bn, &bk, &alpha_one, Q, &bn, v, &one,
               &zero, h_work, &one);
        zgemv_("N", &bn, &bk, &neg_one, Q, &bn, h_work, &one,
               &alpha_one, v, &one);
        v_norm_after = dznrm2_(&bn, v, &one);
    }

    return v_norm_after;
}

/* Legacy MGS kept for backward compatibility, redirects to block version */
double bbjd_mgs_orthogonalize(bbjd_int n, bbjd_int k,
                               const bbjd_zomplex *Q, bbjd_zomplex *v)
{
    /* Allocate small workspace (k is typically small, ≤ maxsub) */
    bbjd_zomplex *h_work = (bbjd_zomplex *)malloc(
        (size_t)(k > 0 ? k : 1) * sizeof(bbjd_zomplex));
    if (!h_work) {
        /* Fallback to BLAS-1 MGS */
        blas_int bn = (blas_int)n, one = 1;
        for (int pass = 0; pass < 2; pass++) {
            for (bbjd_int j = 0; j < k; j++) {
                const bbjd_zomplex *qj = colc(Q, n, j);
                bbjd_zomplex h = zdotc_(&bn, qj, &one, v, &one);
                bbjd_zomplex neg_h = -h;
                zaxpy_(&bn, &neg_h, qj, &one, v, &one);
            }
        }
        return dznrm2_(&bn, v, &one);
    }
    double result = bbjd_block_orthogonalize(n, k, Q, v, h_work);
    free(h_work);
    return result;
}

/* Project v orthogonal to Q[:,0:k] using BLAS-2 */
static void project_out(bbjd_int n, bbjd_int k,
                        const bbjd_zomplex *Q, bbjd_zomplex *v)
{
    if (k == 0) return;

    blas_int bn = (blas_int)n, bk = (blas_int)k, one = 1;
    bbjd_zomplex alpha_one =  1.0;
    bbjd_zomplex neg_one   = -1.0;
    bbjd_zomplex zero      =  0.0;

    /* Stack-allocate small h_work (k typically ≤ nev+few) */
    bbjd_zomplex h_work_stack[128];
    bbjd_zomplex *h_work = h_work_stack;
    bbjd_zomplex *h_work_heap = NULL;
    if (k > 128) {
        h_work_heap = (bbjd_zomplex *)malloc((size_t)k * sizeof(bbjd_zomplex));
        if (!h_work_heap) {
            /* Fallback: BLAS-1 */
            for (int pass = 0; pass < 2; pass++) {
                for (bbjd_int j = 0; j < k; j++) {
                    const bbjd_zomplex *qj = colc(Q, n, j);
                    bbjd_zomplex h = zdotc_(&bn, qj, &one, v, &one);
                    bbjd_zomplex neg_h = -h;
                    zaxpy_(&bn, &neg_h, qj, &one, v, &one);
                }
            }
            return;
        }
        h_work = h_work_heap;
    }

    /* Two passes (always, since k is small here) */
    for (int pass = 0; pass < 2; pass++) {
        zgemv_("C", &bn, &bk, &alpha_one, Q, &bn, v, &one,
               &zero, h_work, &one);
        zgemv_("N", &bn, &bk, &neg_one, Q, &bn, h_work, &one,
               &alpha_one, v, &one);
    }

    if (h_work_heap) free(h_work_heap);
}

/* ================================================================
 * Projected generalized Schur with QZ + reordering
 *
 * 1. Compute QZ decomposition of (MA, MB)
 * 2. Reorder: place ALL locked eigenvalues + the best unconverged
 *    eigenvalue at the top-left block
 * 3. Return the position of the target eigenvalue in *target_pos
 * ================================================================ */
static int solve_projected(bbjd_int k, bbjd_int locked,
                            bbjd_zomplex *MA, bbjd_zomplex *MB,
                            bbjd_zomplex *SA, bbjd_zomplex *SB,
                            bbjd_zomplex *ZA, bbjd_zomplex *ZB,
                            bbjd_zomplex *alpha, bbjd_zomplex *beta,
                            bbjd_zomplex sigma, bbjd_target target,
                            bbjd_int maxsub, bbjd_int nkeep,
                            const bbjd_zomplex *conv_evals,
                            bbjd_int nconv, double tol,
                            bbjd_int *target_pos)
{
    blas_int bk = (blas_int)k, info = 0, lda = (blas_int)maxsub;

    for (bbjd_int j = 0; j < k; j++) {
        memcpy(SA + j*maxsub, MA + j*maxsub, (size_t)k * sizeof(bbjd_zomplex));
        memcpy(SB + j*maxsub, MB + j*maxsub, (size_t)k * sizeof(bbjd_zomplex));
    }

    blas_int sdim = 0;
    double *rwork = malloc((size_t)(8*k) * sizeof(double));
    blas_int *bwork = malloc((size_t)k * sizeof(blas_int));
    if (!rwork || !bwork) { free(rwork); free(bwork); return -1; }

    bbjd_zomplex wq; blas_int lwork = -1;
    zgges_("V","V","N",NULL, &bk, SA, &lda, SB, &lda, &sdim,
           alpha, beta, ZA, &lda, ZB, &lda,
           &wq, &lwork, rwork, bwork, &info);
    if (info) { free(rwork); free(bwork); return -1; }

    lwork = (blas_int)creal(wq) + 1;
    bbjd_zomplex *work = malloc((size_t)lwork * sizeof(bbjd_zomplex));
    if (!work) { free(rwork); free(bwork); return -1; }

    zgges_("V","V","N",NULL, &bk, SA, &lda, SB, &lda, &sdim,
           alpha, beta, ZA, &lda, ZB, &lda,
           work, &lwork, rwork, bwork, &info);
    free(work); free(rwork); free(bwork);
    if (info) return -1;

    /* Build SELECT: mark all eigenvalues matching locked ones + target.
     * This single ZTGSEN call places locked eigenvalues at 0..locked-1
     * and the target at position locked.                              */
    blas_int *sel = calloc(k, sizeof(blas_int));
    if (!sel) return -1;

    /* Mark eigenvalues matching converged ones */
    for (bbjd_int i = 0; i < k; i++) {
        if (cabs(beta[i]) < 1e-300) continue;
        bbjd_zomplex lam = alpha[i] / beta[i];
        for (bbjd_int c = 0; c < nconv; c++) {
            if (cabs(lam - conv_evals[c]) < tol * 100) {
                sel[i] = 1;
                break;
            }
        }
    }

    /* Select the nkeep best unconverged eigenvalues as well, the target
     * among them, so that the leading block of the reordered Schur form is
     * the one a thick restart keeps. */
    for (bbjd_int pick = 0; pick < nkeep; pick++) {
        bbjd_int tidx = -1;
        double best_val = DBL_MAX;
        for (bbjd_int i = 0; i < k; i++) {
            if (sel[i]) continue;
            if (cabs(beta[i]) < 1e-300) continue;
            bbjd_zomplex lam = alpha[i] / beta[i];
            double val;
            switch (target) {
            case BBJD_TARGET_NEAREST:       val =  cabs(lam - sigma); break;
            case BBJD_TARGET_LARGEST_MAG:   val = -cabs(lam); break;
            case BBJD_TARGET_SMALLEST_MAG:  val =  cabs(lam); break;
            case BBJD_TARGET_LARGEST_REAL:  val = -creal(lam); break;
            case BBJD_TARGET_SMALLEST_REAL: val =  creal(lam); break;
            case BBJD_TARGET_LARGEST_IMAG:  val = -cimag(lam); break;
            case BBJD_TARGET_SMALLEST_IMAG: val =  cimag(lam); break;
            default:                        val =  cabs(lam - sigma);
            }
            if (val < best_val) { best_val = val; tidx = i; }
        }
        if (tidx < 0) break;
        sel[tidx] = 1;
    }

    /* Count selected */
    blas_int nsel = 0;
    for (bbjd_int i = 0; i < k; i++) nsel += sel[i];

    if (nsel > 0) {
        blas_int m_out=0, wantq=1, wantz=1, ijob=0;
        double pl, pr, dif[2];
        bbjd_zomplex twq; blas_int tlw=-1, tli=-1; blas_int tiq;
        ztgsen_(&ijob,&wantq,&wantz,sel,&bk,SA,&lda,SB,&lda,
                alpha,beta,ZA,&lda,ZB,&lda,&m_out,&pl,&pr,dif,
                &twq,&tlw,&tiq,&tli,&info);

        tlw = (blas_int)creal(twq)+1; tli = tiq+1;
        bbjd_zomplex *tw = malloc((size_t)tlw*sizeof(bbjd_zomplex));
        blas_int *ti = malloc((size_t)tli*sizeof(blas_int));
        if (!tw||!ti) { free(sel);free(tw);free(ti); return -1; }

        ztgsen_(&ijob,&wantq,&wantz,sel,&bk,SA,&lda,SB,&lda,
                alpha,beta,ZA,&lda,ZB,&lda,&m_out,&pl,&pr,dif,
                tw,&tlw,ti,&tli,&info);
        free(tw); free(ti);
        if (info) { free(sel); return -1; }
    }

    /* Find the target eigenvalue's position.
     * Scan ALL positions for the best unconverged eigenvalue.
     * This is robust regardless of how ZTGSEN ordered the block.  */
    {
        bbjd_int tpos = 0;
        double best_val = DBL_MAX;
        for (bbjd_int i = 0; i < k; i++) {
            if (cabs(beta[i]) < 1e-300) continue;
            bbjd_zomplex lam = alpha[i] / beta[i];
            /* Skip if it matches a converged eigenvalue */
            int is_conv = 0;
            for (bbjd_int c = 0; c < nconv; c++) {
                if (cabs(lam - conv_evals[c]) < tol * 100) {
                    is_conv = 1; break;
                }
            }
            if (is_conv) continue;
            double val;
            switch (target) {
            case BBJD_TARGET_NEAREST:       val =  cabs(lam - sigma); break;
            case BBJD_TARGET_LARGEST_MAG:   val = -cabs(lam); break;
            case BBJD_TARGET_SMALLEST_MAG:  val =  cabs(lam); break;
            case BBJD_TARGET_LARGEST_REAL:  val = -creal(lam); break;
            case BBJD_TARGET_SMALLEST_REAL: val =  creal(lam); break;
            case BBJD_TARGET_LARGEST_IMAG:  val = -cimag(lam); break;
            case BBJD_TARGET_SMALLEST_IMAG: val =  cimag(lam); break;
            default:                        val =  cabs(lam - sigma);
            }
            if (val < best_val) { best_val = val; tpos = i; }
        }
        *target_pos = tpos;
    }

    free(sel);
    return 0;
}

/* ================================================================
 * Stable complex Givens rotation (zlartg-style)
 *
 * Given (f, g), compute real cs >= 0, complex sn, complex r such that
 *
 *   [ cs        conj(sn) ] [ f ]   [ r ]
 *   [ -sn       cs       ] [ g ] = [ 0 ]
 *
 * This is the LAPACK convention: cs is always real and non-negative.
 * Handles |f|=0, |g|=0, and overflow/underflow cases.
 *
 * After rotation, r = cs*f + conj(sn)*g  and  0 = -sn*f + cs*g.
 * ================================================================ */
static void zlartg_stable(bbjd_zomplex f, bbjd_zomplex g,
                          double *cs_out, bbjd_zomplex *sn_out,
                          bbjd_zomplex *r_out)
{
    double af = cabs(f);
    double ag = cabs(g);

    if (ag < 1e-300) {
        /* g is negligible */
        *cs_out = 1.0;
        *sn_out = 0.0;
        *r_out  = f;
        return;
    }
    if (af < 1e-300) {
        /* f is negligible */
        *cs_out = 0.0;
        /* sn = conj(g)/|g| so that conj(sn)*g = |g| is real positive */
        *sn_out = conj(g) / ag;
        *r_out  = ag;
        return;
    }

    double scale = af + ag;  /* avoid overflow */
    double fs = af / scale;
    double gs = ag / scale;
    double norm = scale * sqrt(fs*fs + gs*gs);

    /* cs = |f| / norm (real, non-negative) */
    *cs_out = af / norm;
    /* sn = (f/|f|) * conj(g) / norm */
    *sn_out = (f / af) * conj(g) / norm;
    /* r = (f/|f|) * norm */
    *r_out = (f / af) * norm;
}

/* Column j of the deflation projectors: u orthonormalized against the
 * first j columns of Q_right, its test image nu0 A u + mu0 B u against
 * those of Q_left.                                                   */
static void deflation_append(bbjd_int n, bbjd_int j, bbjd_workspace *ws,
                             const bbjd_zomplex *u, const bbjd_zomplex *Au,
                             const bbjd_zomplex *Bu, bbjd_zomplex nu0, bbjd_zomplex mu0)
{
    blas_int bn = (blas_int)n, one = 1;
    bbjd_zomplex *qr = col(ws->Q_right, n, j), *ql = col(ws->Q_left, n, j);
    zcopy_(&bn, u, &one, qr, &one);
    double nrm = bbjd_block_orthogonalize(n, j, ws->Q_right, qr, ws->h_work);
    if (nrm > 1e-300) { bbjd_zomplex s = 1.0 / nrm; zscal_(&bn, &s, qr, &one); }
    for (bbjd_int i = 0; i < n; i++) ql[i] = nu0 * Au[i] + mu0 * Bu[i];
    nrm = bbjd_block_orthogonalize(n, j, ws->Q_left, ql, ws->h_work);
    if (nrm > 1e-300) { bbjd_zomplex s = 1.0 / nrm; zscal_(&bn, &s, ql, &one); }
}

/* ================================================================
 * GMRES for the JD correction equation (LEFT-preconditioned)
 *
 * Solves the left-preconditioned projected correction equation:
 *
 *   M^{-1} P_L (beta*A - alpha*B) P_R t = -M^{-1} P_L r
 *
 * where:
 *   P_R = I - Q_R Q_R^H  (right projector: eigenvectors)
 *   P_L = I - Q_L Q_L^H  (left projector: B-images)
 *   M ≈ (A - sigma B) is the preconditioner
 *
 * Why left-preconditioning (not right, not FGMRES):
 *   The preconditioned operator M^{-1}(beta*A - alpha*B) has
 *   eigenvalues (beta*lambda - alpha)/(lambda - sigma) ≈
 *   (lambda - theta)/(lambda - sigma).  This clusters all
 *   eigenvalues EXCEPT the target near 1 — ideal for GMRES.
 *
 *   Right preconditioning leaves the raw operator spectrum in
 *   the Krylov space, which is not clustered.  FGMRES adds
 *   flexibility for variable preconditioning but doubles memory.
 *
 * Memory: n × (maxiter+1) for Arnoldi basis V only.
 * No separate Z array (as in FGMRES) needed — left preconditioning
 * keeps solution reconstruction simple: t = V_m * y.
 *
 * Algorithm:
 *   1. r0 = -M^{-1} P_L r    (preconditioned residual)
 *   2. beta0 = ||r0||;  v0 = r0 / beta0
 *   3. For j = 0, ..., m-1:
 *        w = P_R v_j                    (apply right projector)
 *        w = (beta*A - alpha*B) w       (apply operator)
 *        w = P_L w                      (apply left projector)
 *        w = M^{-1} w                   (apply preconditioner)
 *        Arnoldi: H[:,j], v_{j+1}
 *   4. Solve H y = beta0 e0 (least squares via Givens)
 *   5. t = V_m y
 * ================================================================ */
int bbjd_correction_gmres(
    bbjd_int n,
    const bbjd_zomplex *Q_right,
    const bbjd_zomplex *Q_left,
    bbjd_zomplex        theta_a,
    bbjd_zomplex        theta_b,
    const bbjd_zomplex *r,
    bbjd_zomplex       *z,
    bbjd_matvec_fn      matvec_A,
    bbjd_matvec_fn      matvec_B,
    bbjd_precond_fn     precond,
    void               *op_data,
    bbjd_int            maxiter,
    double              tol,
    bbjd_workspace     *ws)
{
    blas_int bn = (blas_int)n, one = 1;
    bbjd_int ims = maxiter;
    bbjd_int ndefl = ws->nlock + 1;

    bbjd_zomplex *V  = ws->gmres_V;   /* Arnoldi basis: n x (ims+1) */
    bbjd_zomplex *H  = ws->gmres_H;
    double       *cs = (double *)ws->gmres_cs;
    bbjd_zomplex *sn = ws->gmres_sn;
    bbjd_zomplex *g  = ws->gmres_g;
    bbjd_zomplex *t1 = ws->tmp3;
    bbjd_zomplex *t2 = ws->tmp4;

    memset(H, 0, (size_t)(ims+1)*ims*sizeof(bbjd_zomplex));

    /* v0 = -M^{-1} P_L r  (preconditioned residual) */
    for (bbjd_int i = 0; i < n; i++) t1[i] = -r[i];
    project_out(n, ndefl, Q_left, t1);

    if (precond) {
        if (precond(n, t1, V, op_data)) return -1;
    } else {
        zcopy_(&bn, t1, &one, V, &one);
    }

    double beta0 = dznrm2_(&bn, V, &one);
    if (beta0 < 1e-300) {
        memset(z, 0, (size_t)n*sizeof(bbjd_zomplex));
        return 0;
    }
    bbjd_zomplex ib = 1.0/beta0;
    zscal_(&bn, &ib, V, &one);

    memset(g, 0, (size_t)(ims+1)*sizeof(bbjd_zomplex));
    g[0] = beta0;

    bbjd_int j;
    for (j = 0; j < ims; j++) {
        bbjd_zomplex *vj  = col(V, n, j);
        bbjd_zomplex *vj1 = col(V, n, j+1);

        /* Apply left-preconditioned operator:
         *   w = M^{-1} P_L (beta*A - alpha*B) P_R v_j                */

        /* t1 = P_R v_j */
        zcopy_(&bn, vj, &one, t1, &one);
        project_out(n, ndefl, Q_right, t1);

        /* t2 = (theta_b*A - theta_a*B) t1 */
        if (matvec_A(n, t1, t2, op_data)) return -1;
        zscal_(&bn, &theta_b, t2, &one);
        if (matvec_B(n, t1, vj1, op_data)) return -1;
        bbjd_zomplex na = -theta_a;
        zaxpy_(&bn, &na, vj1, &one, t2, &one);

        /* t2 = P_L t2 */
        project_out(n, ndefl, Q_left, t2);

        /* vj1 = M^{-1} t2  (preconditioner applied last — left precond) */
        if (precond) {
            if (precond(n, t2, vj1, op_data)) return -1;
        } else {
            zcopy_(&bn, t2, &one, vj1, &one);
        }

        /* Arnoldi: DGKS-monitored MGS */
        double w_norm_before = dznrm2_(&bn, vj1, &one);

        for (bbjd_int i = 0; i <= j; i++) {
            bbjd_zomplex *vi = col(V, n, i);
            bbjd_zomplex h = zdotc_(&bn, vi, &one, vj1, &one);
            H[i + j*(ims+1)] = h;
            bbjd_zomplex nh = -h;
            zaxpy_(&bn, &nh, vi, &one, vj1, &one);
        }
        double w_norm_after = dznrm2_(&bn, vj1, &one);

        if (w_norm_after < 0.7071 * w_norm_before) {
            for (bbjd_int i = 0; i <= j; i++) {
                bbjd_zomplex *vi = col(V, n, i);
                bbjd_zomplex h = zdotc_(&bn, vi, &one, vj1, &one);
                H[i + j*(ims+1)] += h;
                bbjd_zomplex nh = -h;
                zaxpy_(&bn, &nh, vi, &one, vj1, &one);
            }
            w_norm_after = dznrm2_(&bn, vj1, &one);
        }

        double hjj1 = w_norm_after;
        H[(j+1) + j*(ims+1)] = hjj1;

        if (hjj1 < 1e-14 * w_norm_before) { j++; break; }
        bbjd_zomplex ih = 1.0/hjj1;
        zscal_(&bn, &ih, vj1, &one);

        /* Apply previous Givens rotations, G = [cs, sn; -conj(sn), cs]
         * as zlartg_stable defines them (G [f; g] = [r; 0])            */
        for (bbjd_int i = 0; i < j; i++) {
            bbjd_zomplex h1 = H[i   + j*(ims+1)];
            bbjd_zomplex h2 = H[i+1 + j*(ims+1)];
            H[i   + j*(ims+1)] =  cs[i] * h1 + sn[i] * h2;
            H[i+1 + j*(ims+1)] = -conj(sn[i]) * h1 + cs[i] * h2;
        }

        /* New Givens rotation */
        {
            double cs_j;
            bbjd_zomplex sn_j, r_j;
            zlartg_stable(H[j + j*(ims+1)], H[(j+1) + j*(ims+1)],
                          &cs_j, &sn_j, &r_j);
            cs[j] = cs_j;  sn[j] = sn_j;
            H[j   + j*(ims+1)] = r_j;
            H[j+1 + j*(ims+1)] = 0.0;
        }

        /* Apply new rotation to g */
        {
            bbjd_zomplex g1 = g[j], g2 = g[j+1];
            g[j]   =  cs[j] * g1 + sn[j] * g2;
            g[j+1] = -conj(sn[j]) * g1 + cs[j] * g2;
        }

        if (cabs(g[j+1]) < tol * beta0) { j++; break; }
    }
    bbjd_int m = j;

    /* Back-substitution: H[0:m,0:m] y = g[0:m] (upper triangular) */
    bbjd_zomplex *y = g;
    for (bbjd_int i = m-1; i >= 0; i--) {
        for (bbjd_int k2 = i+1; k2 < m; k2++)
            y[i] -= H[i + k2*(ims+1)] * y[k2];
        if (cabs(H[i + i*(ims+1)]) > 1e-300)
            y[i] /= H[i + i*(ims+1)];
    }

    /* Solution: t = V_m y (left-precond: no Z needed) */
    memset(z, 0, (size_t)n*sizeof(bbjd_zomplex));
    for (bbjd_int i = 0; i < m; i++)
        zaxpy_(&bn, &y[i], col(V,n,i), &one, z, &one);

    /* Final right projection: ensure t ⊥ Q_right */
    project_out(n, ndefl, Q_right, z);

    return 0;
}

/* ================================================================
 * Main JDQZ solver — Schur-vector locking
 *
 * After convergence of an eigenvalue, we simply increment nlock.
 * The Schur decomposition from ZGGES+ZTGSEN guarantees that the
 * locked eigenvalues occupy the top-left block, and the Ritz
 * vector at position 'locked' is orthogonal to the locked block
 * by construction of the Schur form.
 *
 * On thick restart, we compress V → V*ZB[:,0:keep] which maps
 * to the Schur basis.  The projected pencil in this new basis
 * is SA[0:keep,0:keep] / SB[0:keep,0:keep] — no dot products
 * needed.
 * ================================================================ */
bbjd_status bbjd_solve(const bbjd_params *params,
                        bbjd_workspace    *ws,
                        bbjd_result       *result)
{
    bbjd_int n       = params->n;
    bbjd_int nev     = params->nev;
    bbjd_int maxsub  = ws->maxsub;
    bbjd_int minsub_auto = 2*nev + 5;
    if (minsub_auto < 20) minsub_auto = 20;
    bbjd_int minsub  = params->minsub > 0 ? params->minsub : minsub_auto;
    bbjd_zomplex sigma = params->sigma;
    bbjd_target target = params->target;
    double tol       = params->tol;
    bbjd_int maxiter = params->maxiter;

    blas_int bn = (blas_int)n, one = 1;
    double nrm;
    bbjd_zomplex inv_nrm;

    result->nconv = 0; result->iters = 0; result->status = BBJD_SUCCESS;
    result->eigenvalues  = calloc(nev, sizeof(bbjd_zomplex));
    result->eigenvectors = calloc((size_t)n*nev, sizeof(bbjd_zomplex));
    result->residuals    = calloc(nev, sizeof(double));
    if (!result->eigenvalues || !result->eigenvectors || !result->residuals)
        { bbjd_result_free(result); return BBJD_ERR_ALLOC; }

    bbjd_zomplex *alpha    = malloc((size_t)maxsub*sizeof(bbjd_zomplex));
    bbjd_zomplex *beta_arr = malloc((size_t)maxsub*sizeof(bbjd_zomplex));
    if (!alpha || !beta_arr)
        { free(alpha); free(beta_arr);
          return BBJD_ERR_ALLOC; }


    /* Test space W = orth(nu0 A V + mu0 B V): for the eigenvalues nearest
     * sigma the harmonic choice (nu0, mu0) = (1, -sigma) / sqrt(1 + |sigma|^2)
     * (Fokkema, Sleijpen & van der Vorst 1998), whose Petrov values avoid
     * the spurious values a Galerkin projection with W = orth(B V) places
     * near an interior target, and which the null rows of a singular B
     * cannot degenerate; nearest 0 is the same with sigma = 0; the other
     * targets keep W = orth(B V).                                        */
    bbjd_zomplex nu0 = 0.0, mu0 = 1.0;
    if (target == BBJD_TARGET_NEAREST) {
        double sc = 1.0 / sqrt(1.0 + cabs(sigma) * cabs(sigma));
        nu0 = sc; mu0 = -sigma * sc;
    } else if (target == BBJD_TARGET_SMALLEST_MAG) {
        nu0 = 1.0; mu0 = 0.0;
    }
    /* ---- Starting vector: params->v0 or random ---- */
    ws->kdim = 0;  ws->nlock = 0;
    bbjd_zomplex *v0 = col(ws->V, n, 0);
    srand(42);
    if (params->v0) {
        memcpy(v0, params->v0, (size_t)n * sizeof(bbjd_zomplex));
    } else {
        for (bbjd_int i = 0; i < n; i++)
            v0[i] = ((double)rand()/RAND_MAX - 0.5)
                  + ((double)rand()/RAND_MAX - 0.5) * I;
    }
    nrm = dznrm2_(&bn, v0, &one);
    if (nrm < 1e-300) { free(alpha); free(beta_arr);
                        result->status = BBJD_ERR_BREAKDOWN; return result->status; }
    inv_nrm = 1.0/nrm;
    zscal_(&bn, &inv_nrm, v0, &one);

    if (params->matvec_A(n, v0, col(ws->AV,n,0), params->op_data))
        goto callback_err;
    if (params->matvec_B(n, v0, col(ws->BV,n,0), params->op_data))
        goto callback_err;

    for (bbjd_int i = 0; i < n; i++)
        col(ws->W,n,0)[i] = nu0 * col(ws->AV,n,0)[i] + mu0 * col(ws->BV,n,0)[i];
    nrm = dznrm2_(&bn, col(ws->W,n,0), &one);
    inv_nrm = 1.0/nrm;
    zscal_(&bn, &inv_nrm, col(ws->W,n,0), &one);

    ws->MA[0] = zdotc_(&bn, col(ws->W,n,0), &one, col(ws->AV,n,0), &one);
    ws->MB[0] = zdotc_(&bn, col(ws->W,n,0), &one, col(ws->BV,n,0), &one);
    ws->kdim = 1;

    /* ---- Main JDQZ iteration ---- */
    for (bbjd_int iter = 0; iter < maxiter; iter++) {
        bbjd_int k = ws->kdim;
        bbjd_int locked = ws->nlock;

        /* Step 1: Solve projected eigenproblem via QZ + reorder */
        bbjd_int target_pos = locked;
        {
            bbjd_int nkeep = minsub - locked;
            if (nkeep < 1) nkeep = 1;
            if (solve_projected(k, locked, ws->MA, ws->MB, ws->SA, ws->SB,
                                ws->ZA, ws->ZB, alpha, beta_arr,
                                sigma, target, maxsub, nkeep,
                                result->eigenvalues, result->nconv, tol,
                                &target_pos))
                { result->status = BBJD_ERR_LAPACK; goto cleanup; }
        }

        /* Step 2: Extract the Ritz pair at the target position.
         *
         * The right generalized Schur vector ZB[:, p] is an eigenvector of
         * the projected pencil only at p = 0; at p > 0 it couples to the
         * leading Schur vectors (the locked pairs among them), and a
         * residual computed with it stalls at the size of that coupling.
         * The eigenvector is ZB[:, 0:p+1] y with y from the triangular
         * pair, (beta SA - alpha SB)[0:p, 0:p] y[0:p] =
         * -(beta SA - alpha SB)[0:p, p] and y[p] = 1, by back-substitution.
         *
         * Ritz value:   theta = alpha[target_pos] / beta[target_pos] */
        bbjd_zomplex theta_a = alpha[target_pos];
        bbjd_zomplex theta_b = beta_arr[target_pos];
        bbjd_zomplex *u  = ws->tmp1;
        {
            bbjd_int p = target_pos;
            bbjd_zomplex *y = ws->h_work;             /* 2 maxsub entries of scratch */
            bbjd_zomplex *x = ws->h_work + maxsub;
            y[p] = 1.0;
            for (bbjd_int i = p - 1; i >= 0; i--) {
                bbjd_zomplex acc = 0.0;
                for (bbjd_int j = i + 1; j <= p; j++)
                    acc += (theta_b * ws->SA[i + j*maxsub]
                            - theta_a * ws->SB[i + j*maxsub]) * y[j];
                bbjd_zomplex d = theta_b * ws->SA[i + i*maxsub]
                               - theta_a * ws->SB[i + i*maxsub];
                y[i] = (cabs(d) > 1e-300) ? -acc / d : 0.0;
            }
            blas_int bk_c = (blas_int)k, bp1 = (blas_int)(p + 1), ldz = (blas_int)maxsub;
            bbjd_zomplex one_c = 1.0, zero_c = 0.0;
            zgemv_("N", &bk_c, &bp1, &one_c, ws->ZB, &ldz, y, &one, &zero_c, x, &one);
            zgemv_("N", &bn, &bk_c, &one_c, ws->V, &bn, x, &one, &zero_c, u, &one);
        }

        nrm = dznrm2_(&bn, u, &one);
        if (nrm < 1e-300) { result->status = BBJD_ERR_BREAKDOWN; goto cleanup; }
        inv_nrm = 1.0/nrm;
        zscal_(&bn, &inv_nrm, u, &one);

        /* Step 3: Compute residual r = theta_b*A*u - theta_a*B*u */
        bbjd_zomplex *Au = ws->tmp2;
        bbjd_zomplex *Bu = ws->tmp3;
        bbjd_zomplex *r  = ws->tmp4;

        if (params->matvec_A(n, u, Au, params->op_data)) goto callback_err;
        if (params->matvec_B(n, u, Bu, params->op_data)) goto callback_err;

        for (bbjd_int i = 0; i < n; i++)
            r[i] = theta_b * Au[i] - theta_a * Bu[i];

        double res_norm = dznrm2_(&bn, r, &one);
        double scale_denom = cabs(theta_a) + cabs(theta_b);
        double rel_res = res_norm / (scale_denom > 1e-300 ? scale_denom : 1.0);

        if (params->verbose >= 2) {
            bbjd_zomplex lam = cabs(theta_b) > 1e-300
                             ? theta_a/theta_b : theta_a;
            fprintf(stderr,
                "  iter %3lld  kdim %3lld  locked %2lld  "
                "eig (%+.6e, %+.6e)  res %.3e\n",
                (long long)iter, (long long)k, (long long)locked,
                creal(lam), cimag(lam), rel_res);
        }

        /* Step 4: Check convergence */
        if (rel_res < tol) {
            bbjd_zomplex lam = cabs(theta_b) > 1e-300
                             ? theta_a/theta_b : theta_a;
            result->eigenvalues[result->nconv] = lam;
            memcpy(col(result->eigenvectors, n, result->nconv),
                   u, (size_t)n*sizeof(bbjd_zomplex));
            result->residuals[result->nconv] = rel_res;
            result->nconv++;

            if (params->verbose >= 1)
                fprintf(stderr,
                    "  *** converged #%lld: (%+.10e, %+.10e)  "
                    "res %.3e  iter %lld\n",
                    (long long)result->nconv,
                    creal(lam), cimag(lam), rel_res, (long long)iter);

            if (result->nconv >= nev) { result->iters = iter+1; goto cleanup; }

            /* the pair stays in the search space; the value keeps it out of
             * the target scan; the vector and its test image join the
             * deflation projectors of the correction equation           */
            deflation_append(n, ws->nlock, ws, u, Au, Bu, nu0, mu0);
            ws->nlock++;

            /* Don't restart — continue with the existing subspace.
             * The Schur decomposition separates locked from active. */
            continue;
        }

        /* Step 6: the correction equation.
         *
         * inner_maxiter > 0: the projected equation
         *   (I - Q_l Q_l^H)(theta_b A - theta_a B)(I - Q_r Q_r^H) t = -r,
         * Q_r = orth[locked eigenvectors, u], Q_l their test images, by
         * left-preconditioned GMRES (an approximate preconditioner).
         *
         * Otherwise the Sleijpen-van der Vorst single step
         *   z = M^{-1} r,  s = M^{-1} u,  t = z - s (u^H z)/(u^H s),
         * with u^H t = 0, exact for the factorization of A - sigma B.    */
        bbjd_zomplex *z_corr = ws->tmp2;
        if (params->inner_maxiter > 0) {
            deflation_append(n, locked, ws, u, Au, Bu, nu0, mu0);
            if (bbjd_correction_gmres(n, ws->Q_right, ws->Q_left, theta_a, theta_b, r, z_corr,
                                      params->matvec_A, params->matvec_B, params->precond,
                                      params->op_data, params->inner_maxiter, params->inner_tol,
                                      ws))
                { result->status = BBJD_ERR_CALLBACK; goto cleanup; }
        } else {
            if (params->precond) {
                if (params->precond(n, r, z_corr, params->op_data))
                    { result->status = BBJD_ERR_CALLBACK; goto cleanup; }
            } else {
                zcopy_(&bn, r, &one, z_corr, &one);
            }
            bbjd_zomplex *s = ws->tmp4;
            if (params->precond) {
                if (params->precond(n, u, s, params->op_data))
                    { result->status = BBJD_ERR_CALLBACK; goto cleanup; }
            } else {
                zcopy_(&bn, u, &one, s, &one);
            }
            bbjd_zomplex u_dot_z = zdotc_(&bn, u, &one, z_corr, &one);
            bbjd_zomplex u_dot_s = zdotc_(&bn, u, &one, s, &one);
            if (cabs(u_dot_s) > 1e-300) {
                bbjd_zomplex coeff = -(u_dot_z / u_dot_s);
                zaxpy_(&bn, &coeff, s, &one, z_corr, &one);
            }
        }

        /* Step 7: Thick restart if subspace is full */
        if (k >= maxsub) {
            bbjd_int keep = minsub < k ? minsub : k-1;
            if (keep < locked+1) keep = locked+1;

            /* Transform to Schur basis: V, AV, BV → V*ZB, AV*ZB, BV*ZB
             *
             * Sequentially use the pre-allocated swap_nm buffer:
             *   zgemm(V, ZB) → swap_nm, then memcpy swap_nm → V
             * Avoids malloc/free on each restart.  Total traffic:
             *   3 × (read V + write swap + read swap + write V) ≈ 4x|V|
             * which is unavoidable for this transformation.           */
            blas_int bk=(blas_int)k, bkeep=(blas_int)keep, lda=(blas_int)maxsub;
            size_t nkeep = (size_t)n * keep;

            zgemm_("N","N",&bn,&bkeep,&bk,&ONE,ws->V,&bn,
                   ws->ZB,&lda,&ZERO,ws->swap_nm,&bn);
            memcpy(ws->V, ws->swap_nm, nkeep*sizeof(bbjd_zomplex));

            zgemm_("N","N",&bn,&bkeep,&bk,&ONE,ws->AV,&bn,
                   ws->ZB,&lda,&ZERO,ws->swap_nm,&bn);
            memcpy(ws->AV, ws->swap_nm, nkeep*sizeof(bbjd_zomplex));

            zgemm_("N","N",&bn,&bkeep,&bk,&ONE,ws->BV,&bn,
                   ws->ZB,&lda,&ZERO,ws->swap_nm,&bn);
            memcpy(ws->BV, ws->swap_nm, nkeep*sizeof(bbjd_zomplex));

            /* Rebuild W by orthogonalizing the test vectors of the kept columns */
            for (bbjd_int j = 0; j < keep; j++) {
                for (bbjd_int i = 0; i < n; i++)
                    col(ws->W,n,j)[i] = nu0 * col(ws->AV,n,j)[i] + mu0 * col(ws->BV,n,j)[i];
                nrm = bbjd_mgs_orthogonalize(n, j, ws->W, col(ws->W,n,j));
                if (nrm > 1e-300) {
                    inv_nrm = 1.0/nrm;
                    zscal_(&bn, &inv_nrm, col(ws->W,n,j), &one);
                }
            }

            /* Projected pencil after restart: compute via BLAS-3 zgemm
             * MA[0:keep, 0:keep] = W[:, 0:keep]^H * AV[:, 0:keep]
             * MB[0:keep, 0:keep] = W[:, 0:keep]^H * BV[:, 0:keep]
             * Much faster than keep^2 individual dot products.           */
            {
                blas_int bkeep = (blas_int)keep;
                blas_int bmaxsub = (blas_int)maxsub;
                bbjd_zomplex one_c = 1.0, zero_c = 0.0;
                zgemm_("C", "N", &bkeep, &bkeep, &bn, &one_c,
                       ws->W, &bn, ws->AV, &bn,
                       &zero_c, ws->MA, &bmaxsub);
                zgemm_("C", "N", &bkeep, &bkeep, &bn, &one_c,
                       ws->W, &bn, ws->BV, &bn,
                       &zero_c, ws->MB, &bmaxsub);
            }

            ws->kdim = keep;
            k = keep;
        }

        /* Step 8: Expand subspace with correction vector */
        bbjd_zomplex *vnew = col(ws->V, n, k);
        zcopy_(&bn, z_corr, &one, vnew, &one);

        nrm = bbjd_block_orthogonalize(n, k, ws->V, vnew, ws->h_work);
        if (nrm < 1e-14) {
            /* Correction too small — inject random vector */
            for (bbjd_int i = 0; i < n; i++)
                vnew[i] = ((double)rand()/RAND_MAX - 0.5)
                         + ((double)rand()/RAND_MAX - 0.5)*I;
            nrm = bbjd_block_orthogonalize(n, k, ws->V, vnew, ws->h_work);
            if (nrm < 1e-14)
                { result->iters=iter+1; result->status=BBJD_ERR_BREAKDOWN;
                  goto cleanup; }
        }
        inv_nrm = 1.0/nrm;
        zscal_(&bn, &inv_nrm, vnew, &one);

        /* Compute AV[:,k] and BV[:,k] */
        if (params->matvec_A(n, vnew, col(ws->AV,n,k), params->op_data))
            goto callback_err;
        if (params->matvec_B(n, vnew, col(ws->BV,n,k), params->op_data))
            goto callback_err;

        /* Compute W[:,k]: the test vector of vnew, orthogonalized against W[:,0:k] */
        bbjd_zomplex *wnew = col(ws->W, n, k);
        for (bbjd_int i = 0; i < n; i++)
            wnew[i] = nu0 * col(ws->AV,n,k)[i] + mu0 * col(ws->BV,n,k)[i];
        nrm = bbjd_block_orthogonalize(n, k, ws->W, wnew, ws->h_work);
        if (nrm < 1e-14) {
            zcopy_(&bn, col(ws->AV,n,k), &one, wnew, &one);
            nrm = bbjd_block_orthogonalize(n, k, ws->W, wnew, ws->h_work);
        }
        if (nrm > 1e-300) {
            inv_nrm = 1.0/nrm;
            zscal_(&bn, &inv_nrm, wnew, &one);
        }

        /* Update projected pencil with new row/column k
         * MA[i,j] = <W[:,i], AV[:,j]> = conj(W[:,i])^T * AV[:,j]
         *
         * Column k: MA[0:k+1, k] = W[:,0:k+1]^H * AV[:,k]
         *   → zgemv("C", W, AV[:,k]) gives sum_p conj(W[p,i]) * AV[p,k]
         *     which is exactly MA[i, k].  Correct.
         *
         * Row k: MA[k, 0:k] = W[:,k]^H * AV[:,0:k]
         *   → we need sum_p conj(W[p,k]) * AV[p,j] for each j.
         *   Computing zgemv("C", AV, W[:,k]) gives
         *     sum_p conj(AV[p,j]) * W[p,k]  =  conj(MA[k,j])
         *   So we'd need to conjugate.  Alternative: just loop zdotc
         *   for the row, which is cheap (k calls, each O(n)).  Since
         *   row-k update is only k dots vs the column which is k+1
         *   dots done as zgemv, the extra cost is negligible.        */
        {
            blas_int bk1 = (blas_int)(k+1);
            bbjd_zomplex one_c = 1.0, zero_c = 0.0;

            /* Column k: BLAS-2 zgemv */
            zgemv_("C", &bn, &bk1, &one_c, ws->W, &bn,
                   col(ws->AV,n,k), &one, &zero_c,
                   ws->MA + k*maxsub, &one);
            zgemv_("C", &bn, &bk1, &one_c, ws->W, &bn,
                   col(ws->BV,n,k), &one, &zero_c,
                   ws->MB + k*maxsub, &one);

            /* Row k: BLAS-1 dots (column k already has MA[k,k])      */
            for (bbjd_int i = 0; i < k; i++) {
                ws->MA[k + i*maxsub] = zdotc_(&bn,
                    col(ws->W,n,k), &one, col(ws->AV,n,i), &one);
                ws->MB[k + i*maxsub] = zdotc_(&bn,
                    col(ws->W,n,k), &one, col(ws->BV,n,i), &one);
            }
        }

        ws->kdim = k+1;
        result->iters = iter+1;
    }

    if (result->nconv < nev) result->status = BBJD_MAX_ITER;

cleanup:
    free(alpha); free(beta_arr);
    return result->status;

callback_err:
    free(alpha); free(beta_arr);
    result->status = BBJD_ERR_CALLBACK;
    return BBJD_ERR_CALLBACK;
}
