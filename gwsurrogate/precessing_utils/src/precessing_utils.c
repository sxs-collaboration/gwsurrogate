#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/numpyconfig.h>
#include <complex.h>

// entire block can be removed if building against numpy 1.X is dropped
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#if defined(__has_include)
#  if __has_include(<numpy/npy_2_compat.h>)
#    include <numpy/npy_2_compat.h>    /* NumPy≥2.0 shim */
#  else
     /* NumPy<2.0: stub so PyArray_ImportNumPyAPI() just calls import_array() */
#    define PyArray_ImportNumPyAPI() (import_array(), 0)
#  endif
#else
#  include <numpy/npy_2_compat.h>      // assume its there
#endif
// END numpy 1.X build support 

#include "precessing_utils.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Created in 2019 by Vijay Varma, Jonathan Blackman
 *
 * This is a utilities Python module for use by gwsurrogate that performs
 * tasks that would be slow in Python.
 *
 * You are free to redistribute and/or modify this software as needed.
 *
 * For help with C extensions for Python including NumPy, see:
 * http://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html
 *
 * The NumPy C-API can be found at:
 * https://docs.scipy.org/doc/numpy/reference/c-api.html
 */

/*
 * Initialize the module in a way compatible with either python 2 or 3. See:
 * https://docs.python.org/3/howto/cporting.html
 */
struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif


/* ==== Setup the python methods table === */
static PyMethodDef _utils_methods[] = {
    {"eval_fit", eval_fit, METH_VARARGS},
    {"normalize_y", normalize_y, METH_VARARGS},
    {"get_ds_fit_x", get_ds_fit_x, METH_VARARGS},
    {"assemble_dydt", assemble_dydt, METH_VARARGS},
    {"ab4_dy", ab4_dy, METH_VARARGS},
    {"binom", binom, METH_VARARGS},
    {"wigner_coef", wigner_coef, METH_VARARGS},
    {"coorbital_to_inertial_in_place", coorbital_to_inertial_in_place, METH_VARARGS},
    {"wignerD_matrices", py_wignerD_matrices, METH_VARARGS},
    {NULL, NULL} /* Marks the end of this structure */
};

#if PY_MAJOR_VERSION >= 3

static int _utils_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int _utils_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_utils",
        NULL,
        sizeof(struct module_state),
        _utils_methods,
        NULL,
        _utils_traverse,
        _utils_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC PyInit__utils(void)

#else

#define INITERROR return

void init_utils(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("_utils", _utils_methods);
#endif
    if (!module)
        INITERROR;

    /* Import NumPy C-API (works on both 1.x and 2.x) */
    if (PyArray_ImportNumPyAPI() < 0) {
        Py_DECREF(module);
        INITERROR;
    }

    //fprintf(stderr,"Build log: Precessing_utils built against NumPy %s\n",NPY_FEATURE_VERSION_STRING);

    struct module_state *st = GETSTATE(module);
    st->error = PyErr_NewException("_utils.Error", NULL, NULL);
    if (!st->error) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

// Done initializing module. Actual functions start here.

double ipow(double base, long exponent) {
    if (exponent == 0) return 1.0;
    double res = base;
    while (exponent > 1) {
        res = res*base;
        exponent -= 1;
    }
    return res;
}

/*
 * This function evaluates a parametric fit.
 * Arguments (with python data types):
 *      bf_orders:  A 2d integer numpy array with shape (n_coefs, 7).
 *                  Gives the basis function orders for each coefficient
 *                  and each parameter.
 *      coefs:      A 1d float numpy array with length n_coefs.
 *                  Contains the fit coefficients.
 *      x: i        A 1d float numpy array with length 7.
 *                  Gives the parameters at which the fit should be evaluated.
 *      q_fit_offset:  A double. Gives the offset for linear transformation
 *                     from q parameter to [-1,1].
 *      q_fit_slope:  A double. Gives the slope for linear transformation
 *                     from q parameter to [-1,1].
 *      q_max_bfOrder: Max basis function order for the q parameter.
 *      chi_max_bfOrder: Max basis function order for the chi parameters.
 *
 * Computes the fit evaluation by summing up coefficients multiplied by 7 basis
 * functions, each evaluated at one component of x.
 * Returns a python float giving the fit evaluation.
 */
static PyObject *eval_fit(PyObject *self, PyObject *args) {

    PyArrayObject *bf_orders, *coefs, *x;
    int i, j, n;
    long *bf_order_data, *orders;
    double res, prod, *coef_data, *x_data, q_fit_offset, q_fit_slope;
    int q_max_bfOrder, chi_max_bfOrder;

    // Parse tuples
    if (!PyArg_ParseTuple(args, "O!O!O!ddii",
            &PyArray_Type, &bf_orders,
            &PyArray_Type, &coefs,
            &PyArray_Type, &x,
            &q_fit_offset,
            &q_fit_slope,
            &q_max_bfOrder,
            &chi_max_bfOrder)) return NULL;

    // array to store powers of fit parameters. These go into the basis
    // functions
    double x_powers[q_max_bfOrder+1 + 6*(chi_max_bfOrder+1)];

    // Point to numpy array data
    bf_order_data = (long *) PyArray_DATA(bf_orders);
    coef_data = (double *) PyArray_DATA(coefs);
    x_data = (double *) PyArray_DATA(x);
    n = PyArray_DIMS(coefs)[0];
    res = 0.0;

    // Compute all needed powers
    for (i=0; i <= q_max_bfOrder; i++){        // power of q parameter
        x_powers[i] = ipow(q_fit_offset + q_fit_slope*x_data[0], i);
    }
    int base_idx;
    for (i=0; i <= chi_max_bfOrder; i++){      // power of chi parameters
        for (j=1; j<7; j++){
            base_idx = q_max_bfOrder+1 + (chi_max_bfOrder+1)*(j-1);
            x_powers[base_idx + i] = ipow(x_data[j], i);
        }
    }

    // Sum up result
    for (i=0; i<n; i++) {
        orders = bf_order_data + i*7;   // shift address of pointer
        prod = x_powers[orders[0]];
        for (j=1; j<7; j++) {
            base_idx = q_max_bfOrder+1 + (chi_max_bfOrder+1)*(j-1);
            prod *= x_powers[base_idx+ orders[j]];
        }
        res += coef_data[i]*prod;
    }

    return Py_BuildValue("d", res);
}


/*
 * This is a helper function to normalize unit quaternions and spin magnitudes
 * at each time step during the ODE integration.
 * Arguments (with python data types):
 *      y:      A 1d float numpy array with length 11, containing:
 *                  y[0:4] is the quaternion
 *                  y[4] is the orbital phase
 *                  y[5:8] is chiA
 *                  y[8:11] is chiB
 *      normA:  A python float giving |chiA|
 *      normB:  A python float giving |chiB|
 * Returns a python float array giving the normalized version of y.
 */
static PyObject *normalize_y(PyObject *self, PyObject *args) {

    PyArrayObject *y, *res;
    int i;
    double *y_data, *res_data, normA, normB, nA, nB, quatNorm, sum;
    npy_intp dims[1];

    // Parse tuples
    if (!PyArg_ParseTuple(args, "O!dd",
            &PyArray_Type, &y,
            &normA,
            &normB)) return NULL;

    // Initialize output array
    dims[0] = 11;
    res = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    res_data = (double *) PyArray_DATA(res);

    y_data = (double *) PyArray_DATA(y);

    // Compute current norms
    sum = 0.0;
    for (i=0; i<4; i++) {
        sum += y_data[i]*y_data[i];
    }
    quatNorm = sqrt(sum);
    sum = 0.0;
    for (i=5; i<8; i++) { // Yes, i=5, not i=4. i=4 is the orbital phase.
        sum += y_data[i]*y_data[i];
    }
    nA = sqrt(sum);
    sum = 0.0;
    for (i=8; i<11; i++) {
        sum += y_data[i]*y_data[i];
    }
    nB = sqrt(sum);

    // Normalize output
    for (i=0; i<4; i++) {
        res_data[i] = y_data[i] / quatNorm;
    }
    res_data[4] = y_data[4];
    for (i=5; i<8; i++) {
        res_data[i] = y_data[i] * normA / nA;
    }
    for (i=8; i<11; i++) {
        res_data[i] = y_data[i] * normB / nB;
    }

    return PyArray_Return(res);
}

/*
 * This function computes the fit input for the dynamics surrogate.
 * Arguments (with python data types):
 *      y:      A 1d float numpy array with length 11, containing:
 *                  y[0:4] is the quaternion
 *                  y[4] is the orbital phase
 *                  y[5:8] is chiA in the coprecessing frame
 *                  y[8:11] is chiB in the coprecessing frame
 *      q:     A python float giving the mass ratio q
 * Returns a python float array x with length 7 containing:
 *                  x[0]: q
 *                  x[1:4]: chiA in the coorbital frame
 *                  x[4:7]: chiB in the coorbital frame
 */
static PyObject *get_ds_fit_x(PyObject *self, PyObject *args) {

    PyArrayObject *y, *x;
    double *y_data, *x_data, q, sp, cp;
    npy_intp dims[1];

    // Parse tuples
    if (!PyArg_ParseTuple(args, "O!d", &PyArray_Type, &y, &q)) return NULL;

    y_data = (double *) PyArray_DATA(y);
    dims[0] = 7;
    x = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    x_data = (double *) PyArray_DATA(x);

    // q
    x_data[0] = q;

    // chiA
    sp = sin(y_data[4]);
    cp = cos(y_data[4]);
    x_data[1] = y_data[5]*cp + y_data[6]*sp;
    x_data[2] = -1*y_data[5]*sp + y_data[6]*cp;
    x_data[3] = y_data[7];

    // chiB
    x_data[4] = y_data[8]*cp + y_data[9]*sp;
    x_data[5] = -1*y_data[8]*sp + y_data[9]*cp;
    x_data[6] = y_data[10];

    return PyArray_Return(x);
}

/*
 * This function assembles the right-hand-side of the dynamics ODE integration
 * Arguments (with python data types):
 *      y:      A 1d float numpy array with length 11, containing:
 *                  y[0:4] is the quaternion
 *                  y[4] is the orbital phase
 *                  y[5:8] is chiA in the coprecessing frame
 *                  y[8:11] is chiB in the coprecessing frame
 *      ooxy:   A length 2 float numpy array with the x and y components of
 *              \Omega^\mathrm{coorb}
 *      omega:  A python float giving the orbital frequency
 *      cAdot:  A length 3 float numpy array with the time derivative of the
 *              coprecessing frame chiA, with components in the coorbital frame.
 *      cBdot:  A length 3 float numpy array with the time derivative of the
 *              coprecessing frame chiB, with components in the coorbital frame.
 * Returns a python float array dydt with length 11 containing the time
 * derivative of y
 */
static PyObject *assemble_dydt(PyObject *self, PyObject *args) {

    PyArrayObject *y, *ooxy, *cAdot, *cBdot, *dydt;
    double *y_data, *ooxy_data, *cAdot_data, *cBdot_data, *dydt_data;
    double omega, sp, cp, ooxy_x, ooxy_y;
    npy_intp dims[1];

    // Parse tuples
    if (!PyArg_ParseTuple(args, "O!O!dO!O!",
            &PyArray_Type, &y,      // length 11, has quat, orbphase, chiA_copr, chiB_copr
            &PyArray_Type, &ooxy,   // length 2 with x and y components of \Omega^{copr}
            &omega,                 // double, orbital frequency \omega
            &PyArray_Type, &cAdot,  // length 3, coprecessing chiA time derivative
            &PyArray_Type, &cBdot)) return NULL;

    // Initialize output array
    dims[0] = 11;
    dydt = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    // Point to numpy array data
    y_data = (double *) PyArray_DATA(y);
    ooxy_data = (double *) PyArray_DATA(ooxy);
    cAdot_data = (double *) PyArray_DATA(cAdot);
    cBdot_data = (double *) PyArray_DATA(cBdot);
    dydt_data = (double *) PyArray_DATA(dydt);

    // Quaternion derivative
    // Omega = 2 * quat^{-1} * dqdt -> dqdt = 0.5 * quat * ooxy_quat where
    // ooxy_quat = [0, ooxy_copr_x, ooxy_copr_y, 0]
    cp = cos(y_data[4]);
    sp = sin(y_data[4]);
    ooxy_x = ooxy_data[0]*cp - ooxy_data[1]*sp;
    ooxy_y = ooxy_data[0]*sp + ooxy_data[1]*cp;
    dydt_data[0] = (-0.5)*y_data[1]*ooxy_x - 0.5*y_data[2]*ooxy_y;
    dydt_data[1] = (-0.5)*y_data[3]*ooxy_y + 0.5*y_data[0]*ooxy_x;
    dydt_data[2] = 0.5*y_data[3]*ooxy_x + 0.5*y_data[0]*ooxy_y;
    dydt_data[3] = 0.5*y_data[1]*ooxy_y - 0.5*y_data[2]*ooxy_x;

    // Orbital phase derivative
    dydt_data[4] = omega;

    // Spin derivatives need to be rotated to the coprecessing frame
    dydt_data[5] = cAdot_data[0]*cp - cAdot_data[1]*sp;
    dydt_data[6] = cAdot_data[0]*sp + cAdot_data[1]*cp;
    dydt_data[7] = cAdot_data[2];
    dydt_data[8] = cBdot_data[0]*cp - cBdot_data[1]*sp;
    dydt_data[9] = cBdot_data[0]*sp + cBdot_data[1]*cp;
    dydt_data[10] = cBdot_data[2];

    return PyArray_Return(dydt);
}

/*
 * A helper function computing the update to y using the Adams-Bashforth
 * 4th-order ODE integration scheme.
 */
static PyObject *ab4_dy(PyObject *self, PyObject *args) {

    PyArrayObject *k1, *k2, *k3, *k4, *res;
    double *k1_data, *k2_data, *k3_data, *k4_data, *res_data;
    double dt1, dt2, dt3, dt4, dt12, dt123, dt23, D1, D2, D3,
            A, B, C, D, B41, B42, B43, B4, C41, C42, C43, C4;
    int i;
    npy_intp dims[1];

    // Parse tuples
    if (!PyArg_ParseTuple(args, "O!O!O!O!dddd",
            &PyArray_Type, &k1,
            &PyArray_Type, &k2,
            &PyArray_Type, &k3,
            &PyArray_Type, &k4,
            &dt1, &dt2, &dt3, &dt4)) return NULL;

    // Initialize output
    dims[0] = 11;
    res = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    // Point to numpy array data
    k1_data = (double *) PyArray_DATA(k1);
    k2_data = (double *) PyArray_DATA(k2);
    k3_data = (double *) PyArray_DATA(k3);
    k4_data = (double *) PyArray_DATA(k4);
    res_data = (double *) PyArray_DATA(res);

    // Various time intervals
    dt12 = dt1 + dt2;
    dt123 = dt12 + dt3;
    dt23 = dt2 + dt3;

    // Denomenators and coefficients
    D1 = dt1 * dt12 * dt123;
    D2 = dt1 * dt2 * dt23;
    D3 = dt2 * dt12 * dt3;

    B41 = dt3 * dt23 / D1;
    B42 = -1 * dt3 * dt123 / D2;
    B43 = dt23 * dt123 / D3;
    B4 = B41 + B42 + B43;

    C41 = (dt23 + dt3) / D1;
    C42 = -1 * (dt123 + dt3) / D2;
    C43 = (dt123 + dt23) / D3;
    C4 = C41 + C42 + C43;

    // Polynomial coefficients and result
    for (i=0; i<11; i++) {
        A = k4_data[i];
        B = k4_data[i]*B4 - k1_data[i]*B41 - k2_data[i]*B42 - k3_data[i]*B43;
        C = k4_data[i]*C4 - k1_data[i]*C41 - k2_data[i]*C42 - k3_data[i]*C43;
        D = (k4_data[i]-k1_data[i])/D1 - (k4_data[i]-k2_data[i])/D2 + (k4_data[i]-k3_data[i])/D3;
        res_data[i] = dt4 * (A + dt4 * (0.5*B + dt4*( C/3.0 + dt4*0.25*D)));
    }

    // Sum up contributions
    return PyArray_Return(res);
}

double factorial(int n) {
    if (n <= 0) return 1.0;
    return factorial(n-1) * n;
}

double factorial_ratio(int n, int k) {
    if (n <= k) return 1.0;
    return factorial_ratio(n-1, k) * n;
}

double _binomial(int n, int k) {
    return factorial_ratio(n, k) / factorial(n-k);
}

double _wigner_coef(int ell, int mp, int m) {
    return sqrt(factorial(ell+m) * factorial(ell-m) / (factorial(ell+mp) * factorial(ell-mp)));
}

static PyObject *binom(PyObject *self, PyObject *args) {
    int n, k;
    double b;

    // Parse tuples
    if (!PyArg_ParseTuple(args, "ii", &n, &k)) return NULL;

    // Compute result
    b = _binomial(n, k);

    // Return result
    return Py_BuildValue("d", b);
}

static PyObject *wigner_coef(PyObject *self, PyObject *args) {
    int ell, mp, m;
    double wc;

    // Parse tuples
    if (!PyArg_ParseTuple(args, "iii", &ell, &mp, &m)) return NULL;

    // Compute result
    wc = _wigner_coef(ell, mp, m);

    // Return result
    return Py_BuildValue("d", wc);
}

/*
 * Co-orbital to inertial in place.
 * For each mode h_lm(t) in h_coorb, multiply by
 *   exp(-i * m * phi_22(t) / 2)
 * where phi_22(t) is the phase of the (2, 2) mode and phi_22 / 2 is the orbital
 * phase.
 * Arguments (with python data types):
 *      h_coorb: A 2d complex numpy array with first index running over (l,m)
 *               modes, and second index running over times. Data is in the
 *               coorbital frame when input, but inertial frame afterwards.
 *      phi_22:  A 1d real numpy array of the phase of the (2,2) mode
 *      m_array: A 1d integer array of the m values of modes, in order
 *
 * Returns None.
 */
static PyObject *coorbital_to_inertial_in_place(PyObject *self, PyObject *args) {

    PyArrayObject *h_coorb, *phi_22, *m_array;

    // Parse tuples
    if (!PyArg_ParseTuple(args, "O!O!O!",
            &PyArray_Type, &h_coorb,  // 2d complex
            &PyArray_Type, &phi_22,   // 1d real
            &PyArray_Type, &m_array)) // 1d int
      return NULL;

    if (PyArray_NDIM(h_coorb) != 2) {
      PyErr_SetString(PyExc_ValueError, "h_coorb must be 2 dimensional");
      return NULL;
    }

    if (PyArray_NDIM(phi_22) != 1) {
      PyErr_SetString(PyExc_ValueError, "phi_22 must be 1 dimensional");
      return NULL;
    }

    if (PyArray_NDIM(m_array) != 1) {
      PyErr_SetString(PyExc_ValueError, "m_array must be 1 dimensional");
      return NULL;
    }

    if (PyArray_TYPE(h_coorb) != NPY_CDOUBLE) {
      PyErr_SetString(PyExc_ValueError, "h_coorb must have dtype complex128");
      return NULL;
    }

    if (PyArray_TYPE(phi_22) != NPY_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "phi_22 must have dtype float64");
      return NULL;
    }

    if (PyArray_TYPE(m_array) != NPY_LONG) {
      PyErr_SetString(PyExc_ValueError, "m_array must have dtype np.long");
      return NULL;
    }

    npy_intp n_modes, n_times;

    n_modes = PyArray_SHAPE(m_array)[0];
    n_times = PyArray_SHAPE(phi_22)[0];

    if ((PyArray_SHAPE(h_coorb)[0] != n_modes) ||
        (PyArray_SHAPE(h_coorb)[1] != n_times)) {
      PyErr_SetString(PyExc_ValueError,
                      "shape incompatibility between h_coorb, phi_22, and m_array");
      return NULL;
    }

    // Figure out the max value of |m|
    npy_long max_m = 0;

    for (npy_long j_m = 0; j_m < n_modes; j_m++) {
      const npy_long this_m = labs(
          *(npy_long *)PyArray_GETPTR1(m_array, j_m));
      max_m = (this_m > max_m) ? this_m : max_m;
    }

    // Storage for all exp's we'll need at a fixed time. They will be ordered
    // by the m number, going from -max_m to +max_m inclusive. For example, if
    // max_m = 5,
    // index:    0  1  2  3  4 5  6  7  8  9 10
    // m value: -5 -4 -3 -2 -1 0 +1 +2 +3 +4 +5
    // So there are a total of 2*max_m+1 elements.
    // Notice that the m=0 element is at twiddle[max_m]. So, a general m value
    // is stored at twiddle[max_m + m].
    double complex *twiddle = malloc((2 * max_m + 1) * sizeof(double complex));

    if (!twiddle) {
      PyErr_SetString(PyExc_ValueError,
                      "Failed to allocate storage for exponential factors.");
      return NULL;
    };

    twiddle[max_m] = CMPLX(1.0, 0.0);

    // Iterate over times the slowest, and modes the fastest, to reuse complex
    // exponential factors
    for (long i_t = 0; i_t < n_times; i_t++) {
      // Precompute all complex exponentials we'll need
      const double phi_22_i = *(double *)PyArray_GETPTR1(phi_22, i_t);
      const double complex exp_minus_i_phi_orb =
          CMPLX(cos(0.5 * phi_22_i), -sin(0.5 * phi_22_i));

      // twiddle[max_m] = 1.0 + 0.0j always
      double complex this_pow = exp_minus_i_phi_orb;
      for (npy_long m = 1; m <= max_m;
           m++, this_pow *= exp_minus_i_phi_orb) {
        twiddle[max_m + m] = this_pow;
        twiddle[max_m - m] = conj(this_pow);
      }

      for (npy_long j_m = 0; j_m < n_modes; j_m++) {
        const npy_long this_m =
            *(npy_long *)PyArray_GETPTR1(m_array, j_m);

        double complex *this_h =
            (double complex *)PyArray_GETPTR2(h_coorb, j_m, i_t);

        *this_h *= twiddle[max_m + this_m];
      };
    };

    free(twiddle);

    Py_RETURN_NONE;
}


/* ------------------------------------------------------------------ */
/*  Bump allocator                                                    */
/* ------------------------------------------------------------------ */

typedef struct { char *ptr, *end; } Bump;

static inline void *bump(Bump *b, size_t bytes)
{
    char *p = (char *)(((size_t)b->ptr + 63) & ~(size_t)63);
    if (p + bytes > b->end) return NULL;
    b->ptr = p + bytes;
    return p;
}

#define BUMP(b, type, count) ((type *)bump(&(b), (size_t)(count) * sizeof(type)))

/* ------------------------------------------------------------------ */
/*  Coefficient helpers — all take the lf table as a parameter        */
/* ------------------------------------------------------------------ */

static inline double rho_coeff(const double *lf,
                               int ell, int mp, int m, int rho)
{
    int a = ell + mp, b = ell - mp, c = ell - rho - m;
    if (rho < 0 || rho > a || c < 0 || c > b) return 0.0;
    double lc = lf[a] - lf[rho] - lf[a - rho]
              + lf[b] - lf[c]   - lf[b - c];
    return (rho & 1 ? -1.0 : 1.0) * exp(lc);
}

static inline double wigner_coef2(const double *lf, int ell, int mp, int m)
{
    return exp(0.5 * (lf[ell+m] + lf[ell-m] - lf[ell+mp] - lf[ell-mp]));
}

/* ------------------------------------------------------------------ */
/*  Complex power table builder                                       */
/* ------------------------------------------------------------------ */

static void build_cpows(const double complex *base,
                        const double complex *inv,
                        size_t len, int half,
                        double complex *out)
{
    size_t z = (size_t)half * len;
    for (size_t k = 0; k < len; ++k) out[z + k] = 1.0;
    for (int p = 1; p <= half; ++p) {
        size_t c = (size_t)(half+p)*len, pv = c - len;
        size_t d = (size_t)(half-p)*len, dv = d + len;
        for (size_t k = 0; k < len; ++k) {
            out[c+k] = out[pv+k] * base[k];
            out[d+k] = out[dv+k] * inv[k];
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Core                                                              */
/* ------------------------------------------------------------------ */

#define W(mat, ell, r, c, t, n) \
    ((mat)[(size_t)(r)*(size_t)(2*(ell)+1)*(n) + (size_t)(c)*(n) + (t)])

int wignerD_matrices(const double *q, size_t n, int ellMax,
                     double complex **matrices)
{
    if (!q || !matrices || ellMax < 2 || n == 0) return -1;

    int NL = ellMax - 1, P = 2 * ellMax, PC = 2*P + 1;
    int lf_size = 2 * ellMax + 2;

    /* ---- Single allocation ---- */
    size_t cap = (size_t)lf_size * sizeof(double)             /* lf table */
      + (size_t)n * (
          2 * sizeof(double complex)                          /* ra_f, rb_f */
        + 3 * sizeof(size_t)                                  /* i1, i2, i3 */
        + 4 * sizeof(double complex)                          /* ra, rb, ia, ib */
        + 2 * (size_t)PC * sizeof(double complex)             /* rapw, rbpw */
        + sizeof(double)                                      /* arSq */
        + (size_t)(P+1) * sizeof(double)                      /* arPw */
        + sizeof(double)                                      /* absR */
        + (size_t)PC * sizeof(double complex)                 /* cpw */
        + sizeof(double complex)                              /* cinv */
      ) + 64 * 24;  /* alignment padding */

    /* Allocate cap + 63 bytes so we can 64-byte-align the usable region.
     * The BumpSizer assumes offset 0 is 64-byte aligned, but malloc may
     * return a pointer with any alignment.  By over-allocating and starting
     * the Bump arena from the first 64-byte-aligned address, the actual
     * layout matches what the sizer computed.  We keep the original malloc
     * pointer (mem) for free(). */
    char *mem = malloc(cap + 63);
    if (!mem) return -1;
    char *base = (char *)(((size_t)mem + 63) & ~(size_t)63);
    Bump B = { base, base + cap };

    /* ---- Log-factorial table: local, no global state ---- */
    double *lf = BUMP(B, double, lf_size);
    if (!lf) { free(mem); return -1; }
    lf[0] = 0.0;
    for (int i = 1; i < lf_size; ++i) lf[i] = lf[i-1] + log((double)i);

    /* ---- Workspace arrays ---- */
    double complex *ra_f = BUMP(B, double complex, n);
    double complex *rb_f = BUMP(B, double complex, n);
    size_t         *idx  = BUMP(B, size_t, 3 * n);
    if (!ra_f || !rb_f || !idx) { free(mem); return -1; }
    size_t *i1 = idx, *i2 = idx + n, *i3 = idx + 2*n;

    const double *q0=q, *q1=q+n, *q2=q+2*n, *q3=q+3*n;
    size_t n1=0, n2=0, n3=0;
    for (size_t j = 0; j < n; ++j) {
        double complex a = q0[j] + I*q3[j], b = q2[j] + I*q1[j];
        ra_f[j] = a; rb_f[j] = b;
        double a2 = creal(a)*creal(a) + cimag(a)*cimag(a);
        double b2 = creal(b)*creal(b) + cimag(b)*cimag(b);
        if      (a2 < 1e-24) i2[n2++] = j;
        else if (b2 < 1e-24) i3[n3++] = j;
        else                  i1[n1++] = j;
    }

    /* ---- Zero matrices only when edge cases present ---- */
    /* General case (i1) writes every element; edge cases only write one     */
    /* column per (ell,m) pair, so off-diagonal entries must start as zero.  */
    if (n2 > 0 || n3 > 0) {
        for (int i = 0; i < NL; ++i) {
            int ell = i + 2;
            size_t bytes = (size_t)(2*ell+1) * (size_t)(2*ell+1) * n
                           * sizeof(double complex);
            memset(matrices[i], 0, bytes);
        }
    }

    /* ---- Edge cases (i2 / i3): reuse same workspace region ---- */
    /* Save bump state so we can reclaim after edge cases */
    char *saved_ptr = B.ptr;

    double complex *cpw  = BUMP(B, double complex, (size_t)PC * n);
    double complex *cinv = BUMP(B, double complex, n);
    if (!cpw || !cinv) { free(mem); return -1; }

    #define DO_EDGE(ni, ia, base_f, row, col, sgn)                      \
    if (ni > 0) {                                                        \
        for (size_t k = 0; k < ni; ++k) cinv[k] = 1.0 / base_f[ia[k]]; \
        for (size_t k = 0; k < ni; ++k) cpw[(size_t)P*ni+k] = 1.0;     \
        for (int p = 1; p <= P; ++p) {                                   \
            size_t c=(size_t)(P+p)*ni, pv=c-ni;                          \
            size_t d=(size_t)(P-p)*ni, dv=d+ni;                          \
            for (size_t k = 0; k < ni; ++k) {                            \
                cpw[c+k] = cpw[pv+k] * base_f[ia[k]];                   \
                cpw[d+k] = cpw[dv+k] * cinv[k];                         \
            }                                                             \
        }                                                                 \
        for (int i = 0; i < NL; ++i) {                                   \
            int ell = i+2; size_t dim = (size_t)(2*ell+1);               \
            for (int m = -ell; m <= ell; ++m) {                           \
                double complex *src = cpw + (size_t)(P+2*m)*ni;          \
                double complex *dst = matrices[i]                         \
                    + (size_t)(row)*dim*n + (size_t)(col)*n;             \
                double s = sgn;                                           \
                for (size_t k = 0; k < ni; ++k) dst[ia[k]] = s*src[k];  \
            }                                                             \
        }                                                                 \
    }

    DO_EDGE(n2, i2, rb_f, ell+m, ell-m, ((ell+m)&1 ? 1.0 : -1.0))
    DO_EDGE(n3, i3, ra_f, ell+m, ell+m, 1.0)
    #undef DO_EDGE

    /* ---- General case (i1) ---- */
    if (n1 > 0) {
        /* Reclaim edge-case workspace */
        B.ptr = saved_ptr;

        double complex *ra   = BUMP(B, double complex, n1);
        double complex *rb   = BUMP(B, double complex, n1);
        double complex *ia   = BUMP(B, double complex, n1);
        double complex *ib   = BUMP(B, double complex, n1);
        double complex *rapw = BUMP(B, double complex, (size_t)PC * n1);
        double complex *rbpw = BUMP(B, double complex, (size_t)PC * n1);
        double         *arSq = BUMP(B, double, n1);
        double         *arPw = BUMP(B, double, (size_t)(P+1) * n1);
        double         *absR = BUMP(B, double, n1);
        if (!ra||!rb||!ia||!ib||!rapw||!rbpw||!arSq||!arPw||!absR)
            { free(mem); return -1; }

        for (size_t k = 0; k < n1; ++k) {
            double complex a = ra_f[i1[k]], b = rb_f[i1[k]];
            ra[k]=a; rb[k]=b; ia[k]=1.0/a; ib[k]=1.0/b;
            double re_a=creal(a), im_a=cimag(a);
            double re_b=creal(b), im_b=cimag(b);
            double a2 = re_a*re_a + im_a*im_a;
            arSq[k] = a2;
            absR[k] = (re_b*re_b + im_b*im_b) / a2;
        }

        build_cpows(ra, ia, n1, P, rapw);
        build_cpows(rb, ib, n1, P, rbpw);

        for (size_t k = 0; k < n1; ++k) arPw[k] = 1.0;
        for (int p = 1; p <= P; ++p) {
            size_t c = (size_t)p*n1, pv = c - n1;
            for (size_t k = 0; k < n1; ++k)
                arPw[c+k] = arPw[pv+k] * arSq[k];
        }

        /* ---- Hot loop: Horner's method ---- */
        for (int i = 0; i < NL; ++i) {
            int ell = i + 2;
            size_t dim = (size_t)(2*ell+1);
            double complex *mat = matrices[i];

            for (int m = -ell; m <= ell; ++m) {
                const double *arSq_row = arPw + (size_t)(ell-m)*n1;

                for (int mp = -ell; mp <= ell; ++mp) {
                    double coef = wigner_coef2(lf, ell, mp, m);
                    int rhoMin = (mp-m > 0) ? (mp-m) : 0;
                    int rhoMax = ell+mp;
                    if (ell-m < rhoMax) rhoMax = ell-m;
                    int nr = rhoMax - rhoMin + 1;

                    double rc[nr]; /* VLA — small, stack-allocated */
                    for (int r = 0; r < nr; ++r)
                        rc[r] = rho_coeff(lf, ell, mp, m, rhoMin + r);

                    const double complex *ar =
                        rapw + (size_t)(P + m+mp)*n1;
                    const double complex *br =
                        rbpw + (size_t)(P + m-mp)*n1;
                    double complex *dst = mat
                        + (size_t)(ell+m)*dim*n + (size_t)(ell+mp)*n;

                    for (size_t k = 0; k < n1; ++k) {
                        double R = absR[k];
                        double s = rc[nr-1];
                        for (int r = nr-2; r >= 0; --r)
                            s = s*R + rc[r];
                        if      (rhoMin == 1) s *= R;
                        else if (rhoMin == 2) s *= R*R;
                        else if (rhoMin > 2) {
                            double Rp = R*R;
                            for (int p = 2; p < rhoMin; ++p) Rp *= R;
                            s *= Rp;
                        }
                        dst[i1[k]] = (coef*s)*ar[k]*br[k]*arSq_row[k];
                    }
                }
            }
        }
    }

    free(mem);
    return 0;
}

PyObject *py_wignerD_matrices(PyObject *self, PyObject *args)
{
    PyArrayObject *q_obj;
    PyObject *mat_list;
    int ellMax;

    if (!PyArg_ParseTuple(args, "O!iO!",
            &PyArray_Type, &q_obj,
            &ellMax,
            &PyList_Type, &mat_list))
        return NULL;

    if (PyArray_NDIM(q_obj) != 2 || PyArray_DIM(q_obj, 0) != 4) {
        PyErr_SetString(PyExc_ValueError, "q must have shape (4, N)");
        return NULL;
    }
    if (ellMax < 2) {
        PyErr_SetString(PyExc_ValueError, "ellMax must be >= 2");
        return NULL;
    }

    int num_ells = ellMax - 1;
    if (PyList_GET_SIZE(mat_list) != num_ells) {
        PyErr_Format(PyExc_ValueError,
                     "matrices list must have length %d", num_ells);
        return NULL;
    }

    PyArrayObject *q_arr = (PyArrayObject *)PyArray_ContiguousFromAny(
        (PyObject *)q_obj, NPY_DOUBLE, 2, 2);
    if (!q_arr) return NULL;

    npy_intp n = PyArray_DIM(q_arr, 1);
    const double *q_data = (const double *)PyArray_DATA(q_arr);

    /* Stack-allocated pointer array — no malloc needed.
     * num_ells = ellMax - 1, so for any reasonable ellMax this is tiny. */
    double complex *mat_ptrs[num_ells];

    for (int i = 0; i < num_ells; ++i) {
        int ell = i + 2;
        int dim = 2 * ell + 1;
        PyObject *item = PyList_GET_ITEM(mat_list, i);

        if (!PyArray_Check(item)) {
            PyErr_Format(PyExc_TypeError,
                         "matrices[%d] is not a numpy array", i);
            Py_DECREF(q_arr);
            return NULL;
        }

        PyArrayObject *mat = (PyArrayObject *)item;

        if (PyArray_NDIM(mat) != 3 ||
            PyArray_DIM(mat, 0) != dim ||
            PyArray_DIM(mat, 1) != dim ||
            PyArray_DIM(mat, 2) != n) {
            PyErr_Format(PyExc_ValueError,
                         "matrices[%d] must have shape (%d, %d, %ld)",
                         i, dim, dim, (long)n);
            Py_DECREF(q_arr);
            return NULL;
        }
        if (PyArray_TYPE(mat) != NPY_COMPLEX128) {
            PyErr_Format(PyExc_TypeError,
                         "matrices[%d] must have dtype complex128", i);
            Py_DECREF(q_arr);
            return NULL;
        }
        if (!PyArray_IS_C_CONTIGUOUS(mat)) {
            PyErr_Format(PyExc_ValueError,
                         "matrices[%d] must be C-contiguous", i);
            Py_DECREF(q_arr);
            return NULL;
        }

        /* Direct pointer into NumPy's buffer — no copy */
        mat_ptrs[i] = (double complex *)PyArray_DATA(mat);
    }

    int ret;
    Py_BEGIN_ALLOW_THREADS
    ret = wignerD_matrices(q_data, (size_t)n, ellMax, mat_ptrs);
    Py_END_ALLOW_THREADS

    Py_DECREF(q_arr);

    if (ret != 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "wignerD_matrices failed (internal error)");
        return NULL;
    }

    Py_RETURN_NONE;
}
