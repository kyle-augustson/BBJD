/*
 * _bbjd_c.c — Python C extension for the BBJD eigensolver
 *
 * Provides a single Python-callable function:
 *
 *   jdqz_c(n, nev, sigma, target,
 *          matvec_A, matvec_B, precond_solve,
 *          tol, maxiter, maxsub, minsub,
 *          inner_maxiter, inner_tol, verbose, v0)
 *
 * matvec_A, matvec_B: callable(x) -> y   (numpy arrays)
 * precond_solve:      callable(x) -> y   or None
 * v0:                 complex128 array of length n, or None (random)
 *
 * Returns (eigenvalues, eigenvectors, residuals, nconv, iters, status)
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include "bbjd.h"

/* ================================================================
 * Callback trampoline data: holds the Python callables
 * ================================================================ */
typedef struct {
    PyObject *matvec_A;
    PyObject *matvec_B;
    PyObject *precond;
    npy_intp  n;
    int       error;        /* set nonzero if a Python call fails */
} CallbackData;

/* ================================================================
 * Callback trampolines: C -> Python -> C
 *
 * These are called by the C solver.  They wrap the input vector
 * as a numpy array, call the Python callable, and copy the result
 * back into the output vector.
 * ================================================================ */
static int trampoline_matvec_A(bbjd_int n, const bbjd_zomplex *x,
                                bbjd_zomplex *y, void *data)
{
    CallbackData *cb = (CallbackData *)data;
    if (cb->error) return -1;

    /* Wrap x as a numpy array (non-owning view) */
    npy_intp dims = (npy_intp)n;
    PyObject *x_arr = PyArray_SimpleNewFromData(
        1, &dims, NPY_COMPLEX128, (void *)x);
    if (!x_arr) { cb->error = 1; return -1; }
    /* Mark read-only */
    PyArray_CLEARFLAGS((PyArrayObject *)x_arr, NPY_ARRAY_WRITEABLE);

    /* Call Python */
    PyObject *result = PyObject_CallFunctionObjArgs(cb->matvec_A, x_arr, NULL);
    Py_DECREF(x_arr);
    if (!result) { cb->error = 1; return -1; }

    /* Extract result into y */
    PyArrayObject *y_arr = (PyArrayObject *)PyArray_FROM_OTF(
        result, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    Py_DECREF(result);
    if (!y_arr) { cb->error = 1; return -1; }

    memcpy(y, PyArray_DATA(y_arr), (size_t)n * sizeof(bbjd_zomplex));
    Py_DECREF(y_arr);
    return 0;
}

static int trampoline_matvec_B(bbjd_int n, const bbjd_zomplex *x,
                                bbjd_zomplex *y, void *data)
{
    CallbackData *cb = (CallbackData *)data;
    if (cb->error) return -1;

    npy_intp dims = (npy_intp)n;
    PyObject *x_arr = PyArray_SimpleNewFromData(
        1, &dims, NPY_COMPLEX128, (void *)x);
    if (!x_arr) { cb->error = 1; return -1; }
    PyArray_CLEARFLAGS((PyArrayObject *)x_arr, NPY_ARRAY_WRITEABLE);

    PyObject *result = PyObject_CallFunctionObjArgs(cb->matvec_B, x_arr, NULL);
    Py_DECREF(x_arr);
    if (!result) { cb->error = 1; return -1; }

    PyArrayObject *y_arr = (PyArrayObject *)PyArray_FROM_OTF(
        result, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    Py_DECREF(result);
    if (!y_arr) { cb->error = 1; return -1; }

    memcpy(y, PyArray_DATA(y_arr), (size_t)n * sizeof(bbjd_zomplex));
    Py_DECREF(y_arr);
    return 0;
}

static int trampoline_precond(bbjd_int n, const bbjd_zomplex *x,
                               bbjd_zomplex *y, void *data)
{
    CallbackData *cb = (CallbackData *)data;
    if (cb->error) return -1;

    npy_intp dims = (npy_intp)n;
    PyObject *x_arr = PyArray_SimpleNewFromData(
        1, &dims, NPY_COMPLEX128, (void *)x);
    if (!x_arr) { cb->error = 1; return -1; }
    PyArray_CLEARFLAGS((PyArrayObject *)x_arr, NPY_ARRAY_WRITEABLE);

    PyObject *result = PyObject_CallFunctionObjArgs(cb->precond, x_arr, NULL);
    Py_DECREF(x_arr);
    if (!result) { cb->error = 1; return -1; }

    PyArrayObject *y_arr = (PyArrayObject *)PyArray_FROM_OTF(
        result, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    Py_DECREF(result);
    if (!y_arr) { cb->error = 1; return -1; }

    memcpy(y, PyArray_DATA(y_arr), (size_t)n * sizeof(bbjd_zomplex));
    Py_DECREF(y_arr);
    return 0;
}

/* ================================================================
 * Python entry point
 * ================================================================ */
static PyObject *
py_jdqz(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;

    Py_ssize_t n_sst, nev_sst;
    Py_complex sigma_py;
    int target_int;
    PyObject *mvA_obj, *mvB_obj, *pc_obj, *v0_obj = Py_None;
    double tol = 1e-10, inner_tol = 1e-2;
    Py_ssize_t maxiter_sst = 200, maxsub_sst = 0, minsub_sst = 0, inner_maxiter_sst = 0;
    int verbose = 0;

    static char *kwlist[] = {
        "n", "nev", "sigma", "target",
        "matvec_A", "matvec_B", "precond_solve",
        "tol", "maxiter", "maxsub", "minsub",
        "inner_maxiter", "inner_tol", "verbose", "v0",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "nnDiOOO|dnnnndiO", kwlist,
            &n_sst, &nev_sst, &sigma_py, &target_int,
            &mvA_obj, &mvB_obj, &pc_obj,
            &tol, &maxiter_sst, &maxsub_sst, &minsub_sst,
            &inner_maxiter_sst, &inner_tol, &verbose, &v0_obj))
        return NULL;

    bbjd_int n   = (bbjd_int)n_sst;
    bbjd_int nev = (bbjd_int)nev_sst;

    PyArrayObject *v0_arr = NULL;
    if (v0_obj != Py_None) {
        v0_arr = (PyArrayObject *)PyArray_FROM_OTF(v0_obj, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
        if (!v0_arr) return NULL;
        if (PyArray_NDIM(v0_arr) != 1 || PyArray_DIM(v0_arr, 0) != (npy_intp)n) {
            Py_DECREF(v0_arr);
            return PyErr_Format(PyExc_ValueError, "v0 must have shape (%ld,)", (long)n);
        }
    }

    /* Set up callbacks */
    CallbackData cb;
    cb.matvec_A = mvA_obj;
    cb.matvec_B = mvB_obj;
    cb.precond  = (pc_obj == Py_None) ? NULL : pc_obj;
    cb.n        = (npy_intp)n;
    cb.error    = 0;

    /* Set up params */
    bbjd_params params;
    bbjd_params_init(&params);
    params.n             = n;
    params.nev           = nev;
    params.sigma         = sigma_py.real + sigma_py.imag * I;
    params.target        = (bbjd_target)target_int;
    params.tol           = tol;
    params.maxiter       = (bbjd_int)maxiter_sst;
    params.maxsub        = (bbjd_int)maxsub_sst;
    params.minsub        = (bbjd_int)minsub_sst;
    params.inner_maxiter = (bbjd_int)inner_maxiter_sst;
    params.inner_tol     = inner_tol;
    params.matvec_A      = trampoline_matvec_A;
    params.matvec_B      = trampoline_matvec_B;
    params.precond       = cb.precond ? trampoline_precond : NULL;
    params.op_data       = &cb;
    params.verbose       = verbose;
    params.v0            = v0_arr ? (const bbjd_zomplex *)PyArray_DATA(v0_arr) : NULL;

    /* Allocate workspace */
    bbjd_workspace *ws = bbjd_workspace_alloc(&params);
    if (!ws) { Py_XDECREF(v0_arr); return PyErr_NoMemory(); }

    /* Run solver */
    bbjd_result result;
    memset(&result, 0, sizeof(result));

    bbjd_status status = bbjd_solve(&params, ws, &result);

    bbjd_workspace_free(ws);
    Py_XDECREF(v0_arr);

    /* Check for Python errors from callbacks */
    if (cb.error) {
        bbjd_result_free(&result);
        /* Python exception already set by the trampoline */
        return NULL;
    }

    /* Build return tuple */
    bbjd_int nconv = result.nconv;
    npy_intp ev_dims[1]  = { (npy_intp)nconv };
    npy_intp vec_dims[2] = { (npy_intp)n, (npy_intp)nconv };
    npy_intp res_dims[1] = { (npy_intp)nconv };

    PyObject *evals = PyArray_SimpleNew(1, ev_dims, NPY_COMPLEX128);
    PyObject *evecs = PyArray_SimpleNew(2, vec_dims, NPY_COMPLEX128);
    PyObject *resid = PyArray_SimpleNew(1, res_dims, NPY_FLOAT64);

    if (!evals || !evecs || !resid) {
        Py_XDECREF(evals); Py_XDECREF(evecs); Py_XDECREF(resid);
        bbjd_result_free(&result);
        return PyErr_NoMemory();
    }

    if (nconv > 0) {
        memcpy(PyArray_DATA((PyArrayObject *)evals),
               result.eigenvalues,
               (size_t)nconv * sizeof(bbjd_zomplex));
        /* Eigenvectors are stored column-major in C (Fortran order).
         * NumPy default is row-major.  Copy column by column.       */
        bbjd_zomplex *evec_data = (bbjd_zomplex *)PyArray_DATA(
            (PyArrayObject *)evecs);
        for (bbjd_int j = 0; j < nconv; j++) {
            for (bbjd_int i = 0; i < n; i++) {
                evec_data[i * nconv + j] = result.eigenvectors[j * n + i];
            }
        }
        memcpy(PyArray_DATA((PyArrayObject *)resid),
               result.residuals,
               (size_t)nconv * sizeof(double));
    }

    bbjd_result_free(&result);

    return Py_BuildValue("OOOnni",
                         evals, evecs, resid,
                         (Py_ssize_t)nconv,
                         (Py_ssize_t)result.iters,
                         (int)status);
}

/* ================================================================
 * Module definition
 * ================================================================ */
static PyMethodDef BbjdMethods[] = {
    {"jdqz_c", (PyCFunction)py_jdqz, METH_VARARGS | METH_KEYWORDS,
     "jdqz_c(n, nev, sigma, target,\n"
     "        matvec_A, matvec_B, precond_solve,\n"
     "        tol=1e-10, maxiter=200, maxsub=0, minsub=0,\n"
     "        inner_maxiter=0, inner_tol=0.01, verbose=0, v0=None)\n"
     "\n"
     "Jacobi-Davidson QZ eigensolver for A x = lambda B x.\n"
     "Returns (eigenvalues, eigenvectors, residuals, nconv, iters, status).\n"
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_bbjd_c",
    "Block-Bordered Jacobi-Davidson eigensolver (C extension)",
    -1,
    BbjdMethods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__bbjd_c(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
