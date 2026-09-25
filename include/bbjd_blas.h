/*
 * bbjd_blas.h — BLAS/LAPACK prototypes used by the eigensolver
 *
 * We call Fortran BLAS/LAPACK via the standard C name-mangling
 * convention (lowercase with trailing underscore).
 *
 * When linking against scipy's bundled OpenBLAS, symbols are prefixed
 * with "scipy_".  Define BBJD_SCIPY_OPENBLAS to use prefixed names.
 */
#ifndef BBJD_BLAS_H
#define BBJD_BLAS_H

#include <complex.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* We use int for BLAS/LAPACK integer (matches Fortran default INTEGER).
 * scipy's bundled OpenBLAS uses 32-bit integers.                       */
typedef int blas_int;

/* ================================================================
 * Symbol name mapping
 *
 * When BBJD_SCIPY_OPENBLAS is defined, all BLAS/LAPACK symbols get
 * the "scipy_" prefix.  Otherwise use standard Fortran names.
 * ================================================================ */
#ifdef BBJD_SCIPY_OPENBLAS
#define BLAS(name) scipy_ ## name
#else
#define BLAS(name) name
#endif

/* ================================================================
 * BLAS Level 1
 * ================================================================ */

extern void BLAS(zaxpy_)(const blas_int *n,
                         const double complex *alpha,
                         const double complex *x, const blas_int *incx,
                         double complex *y, const blas_int *incy);

extern void BLAS(zscal_)(const blas_int *n,
                         const double complex *alpha,
                         double complex *x, const blas_int *incx);

extern void BLAS(zcopy_)(const blas_int *n,
                         const double complex *x, const blas_int *incx,
                         double complex *y, const blas_int *incy);

/* zdotc through the CBLAS "_sub" form: complex-valued Fortran functions
 * return by value (gfortran) or through a hidden pointer (Intel/MKL), and a
 * process that already holds MKL resolves zdotc_ to MKL's copy.  The CBLAS
 * form has one C ABI in every implementation.                            */
extern void BLAS(cblas_zdotc_sub)(const blas_int n,
                                  const void *x, const blas_int incx,
                                  const void *y, const blas_int incy,
                                  void *dotc);

static inline double complex bbjd_zdotc(const blas_int *n,
                                        const double complex *x, const blas_int *incx,
                                        const double complex *y, const blas_int *incy)
{
    double complex r;
    BLAS(cblas_zdotc_sub)(*n, x, *incx, y, *incy, &r);
    return r;
}

extern double BLAS(dznrm2_)(const blas_int *n,
                            const double complex *x, const blas_int *incx);

/* ================================================================
 * BLAS Level 2
 * ================================================================ */

extern void BLAS(zgemv_)(const char *trans,
                         const blas_int *m, const blas_int *n,
                         const double complex *alpha,
                         const double complex *A, const blas_int *lda,
                         const double complex *x, const blas_int *incx,
                         const double complex *beta,
                         double complex *y, const blas_int *incy);

/* ================================================================
 * BLAS Level 3
 * ================================================================ */

extern void BLAS(zgemm_)(const char *transa, const char *transb,
                         const blas_int *m, const blas_int *n, const blas_int *k,
                         const double complex *alpha,
                         const double complex *A, const blas_int *lda,
                         const double complex *B, const blas_int *ldb,
                         const double complex *beta,
                         double complex *C, const blas_int *ldc);

/* ================================================================
 * LAPACK
 * ================================================================ */

typedef blas_int (*lapack_selctg)(const double complex *,
                                  const double complex *);

extern void BLAS(zgges_)(const char *jobvsl, const char *jobvsr,
                         const char *sort, lapack_selctg selctg,
                         const blas_int *n,
                         double complex *A, const blas_int *lda,
                         double complex *B, const blas_int *ldb,
                         blas_int *sdim,
                         double complex *alpha, double complex *beta,
                         double complex *vsl, const blas_int *ldvsl,
                         double complex *vsr, const blas_int *ldvsr,
                         double complex *work, const blas_int *lwork,
                         double *rwork, blas_int *bwork,
                         blas_int *info);

extern void BLAS(ztgsen_)(const blas_int *ijob,
                          const blas_int *wantq, const blas_int *wantz,
                          const blas_int *select, const blas_int *n,
                          double complex *A, const blas_int *lda,
                          double complex *B, const blas_int *ldb,
                          double complex *alpha, double complex *beta,
                          double complex *Q, const blas_int *ldq,
                          double complex *Z, const blas_int *ldz,
                          blas_int *m,
                          double *pl, double *pr, double *dif,
                          double complex *work, const blas_int *lwork,
                          blas_int *iwork, const blas_int *liwork,
                          blas_int *info);

extern void BLAS(ztrsv_)(const char *uplo, const char *trans, const char *diag,
                         const blas_int *n,
                         const double complex *A, const blas_int *lda,
                         double complex *x, const blas_int *incx);

extern void BLAS(zgesv_)(const blas_int *n, const blas_int *nrhs,
                         double complex *A, const blas_int *lda,
                         blas_int *ipiv,
                         double complex *B, const blas_int *ldb,
                         blas_int *info);

/* ================================================================
 * Convenience macros: call BLAS/LAPACK without prefix noise
 * ================================================================ */
#define zaxpy_  BLAS(zaxpy_)
#define zscal_  BLAS(zscal_)
#define zcopy_  BLAS(zcopy_)
#define zdotc_  bbjd_zdotc
#define dznrm2_ BLAS(dznrm2_)
#define zgemv_  BLAS(zgemv_)
#define zgemm_  BLAS(zgemm_)
#define zgges_  BLAS(zgges_)
#define ztgsen_ BLAS(ztgsen_)
#define ztrsv_  BLAS(ztrsv_)
#define zgesv_  BLAS(zgesv_)

#ifdef __cplusplus
}
#endif

#endif /* BBJD_BLAS_H */
