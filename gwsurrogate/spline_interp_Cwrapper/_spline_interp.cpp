/**
 * _spline_interp.cpp — Fast natural cubic spline interpolation
 *
 * C++14 implementation.  Exposes the same extern "C" ABI as the original C
 * file so the Python ctypes wrapper (spline_interp_Cwrapper.py) is unchanged.
 *
 * Design goals met:
 *   - Single heap allocation per prepare/evaluate call
 *   - FMA-friendly evaluation via Horner's method
 *   - Precomputed reciprocals for inner loops
 *   - Multi-dataset support sharing one spline_plan
 *   - restrict-annotated pointers throughout
 *   - GCC/Clang vectorization pragmas
 *   - Thread-safe (no global/static state)
 *   - Hunt algorithm for O(log n) → O(1) amortized lookup on sorted out_x
 *
 * The real (`double`) and complex (`complex128`, i.e. interleaved re/im pairs)
 * multi-dataset paths share a single template implementation
 * `spline_interp_multi_tmpl<T>`.  The extern "C" wrappers reinterpret-cast
 * the complex double** arguments to std::complex<double>** — safe by C++11
 * §26.4 which guarantees [re, im] memory layout identical to double[2].
 */

#include <Python.h>
#include <cassert>
#include <cmath>      /* std::fma                    */
#include <complex>    /* std::complex<double>        */
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>     /* std::unique_ptr             */
#include <new>        /* std::bad_alloc              */
#include <vector>     /* std::vector<unsigned char>  */

/* -------------------------------------------------------------------------
 * restrict is a C99 keyword; GCC and Clang support __restrict__ in C++.
 * ---------------------------------------------------------------------- */
#define restrict __restrict__

/* -------------------------------------------------------------------------
 * Return codes
 * ---------------------------------------------------------------------- */
#define SPLINE_OK            0
#define SPLINE_ERR_OOM       1   /* allocation failed                      */
#define SPLINE_ERR_BOUNDS    2   /* out_x value outside [x0, x_{n-1}]      */
#define SPLINE_ERR_SINGULAR  3   /* zero-length interval in data_x         */
#define SPLINE_ERR_DATA_SIZE 4   /* data_size < 3                          */
#define SPLINE_ERR_OUT_SIZE  5   /* out_size <= 0                          */
#define SPLINE_ERR_NUM_DS    6   /* num_datasets <= 0                      */

/* -------------------------------------------------------------------------
 * Branch-prediction hints and always-inline
 * ---------------------------------------------------------------------- */
#if defined(__GNUC__) || defined(__clang__)
#  define SPLINE_LIKELY(x)   __builtin_expect(!!(x), 1)
#  define SPLINE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#  define SPLINE_INLINE      __attribute__((always_inline)) static inline
#else
#  define SPLINE_LIKELY(x)   (x)
#  define SPLINE_UNLIKELY(x) (x)
#  define SPLINE_INLINE      static inline
#endif

/* -------------------------------------------------------------------------
 * spline_fma — fused multiply-add, type-safe.
 *
 * For double: uses hardware std::fma() (exact; emits FMA instruction when
 *             the target supports it).
 * For std::complex<double>: uses a*b + c (FMA undefined for complex).
 *
 * Three overloads cover all mixed-scalar patterns in the spline kernel:
 *   spline_fma(double, double, double) — non-template, preferred by overload
 *                                        resolution for the all-double path
 *   spline_fma(T,      double, T     ) — RHS computation: coef * inv_h + acc
 *   spline_fma(double, T,      T     ) — Horner evaluation: scalar * poly + c
 * ---------------------------------------------------------------------- */
static inline double spline_fma(double a, double b, double c) {
    return std::fma(a, b, c);
}

template<typename T>
static inline T spline_fma(T a, double b, T c) { return a * b + c; }

template<typename T>
static inline T spline_fma(double a, T b, T c) { return a * b + c; }

/* ──────────────────────────────────────────────
 * Hunt algorithm for interval location.
 *
 * Given a target value `target`, find the interval index `idx` such
 * that data_x[idx] <= target <= data_x[idx + 1].
 *
 * Starts from `guess` (typically the last known interval) and
 * expands geometrically until the target is bracketed, then
 * binary searches within the bracket.
 *
 * Returns the interval index, or -1 if target is out of range.
 *
 * Reference: Numerical Recipes, "hunt" routine.
 * ────────────────────────────────────────────── */
static int hunt(const double *const restrict data_x,
                const int data_size,
                const double target,
                const int guess)
{
    const int last_interval = data_size - 2;

    /* Out of range checks */
    if (target < data_x[0] || target > data_x[data_size - 1]) {
        return -1;
    }

    /* Clamp guess to valid range */
    int lo, hi;
    int g = guess;
    if (g < 0) g = 0;
    if (g > last_interval) g = last_interval;

    /* Check if we're already in the right interval */
    if (data_x[g] <= target && target <= data_x[g + 1]) {
        return g;
    }

    /* Determine hunt direction */
    if (target >= data_x[g]) {
        /* Hunt upward */
        lo = g;
        int increment = 1;
        hi = lo + increment;
        while (hi <= last_interval && data_x[hi] < target) {
            lo = hi;
            increment *= 2;
            hi = lo + increment;
        }
        if (hi > last_interval) {
            hi = last_interval;
        }
    } else {
        /* Hunt downward */
        hi = g;
        int increment = 1;
        lo = hi - increment;
        while (lo >= 0 && data_x[lo + 1] > target) {
            hi = lo;
            increment *= 2;
            lo = hi - increment;
        }
        if (lo < 0) {
            lo = 0;
        }
    }

    /* Binary search within [lo, hi] */
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (data_x[mid] <= target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    return lo;
}

/* =========================================================================
 * spline_plan (internal definition)
 *
 * Owns one heap allocation containing:
 *   double h[n-1]          — interval widths  data_x[i+1] - data_x[i]
 *   double inv_h[n-1]      — 1/h[i], precomputed
 *   double diag[n-2]       — factored interior diagonal (Thomas algorithm)
 *   (data_x pointer is stored but NOT owned — caller keeps it alive)
 * ====================================================================== */
struct spline_plan {
    const double *data_x;   /* NOT owned */
    long          n;

    void   *mem;            /* single heap block               */
    double *h;              /* h[n-1]   interval widths        */
    double *inv_h;          /* inv_h[n-1] = 1/h[i]             */
    double *diag;           /* diag[n-2] factored interior diag */
};

static size_t align_double(size_t nb) {
    const size_t a = sizeof(double);
    return (nb + a - 1u) & ~(a - 1u);
}

/* =========================================================================
 * spline_prepare — compute geometry arrays from data_x.
 *
 * Returns a heap-allocated spline_plan on success, NULL on failure.
 * Caller must free with spline_plan_free().
 * data_x must remain valid for the lifetime of the plan.
 *
 * Natural spline: y''(x_0) = y''(x_{n-1}) = 0.
 *
 * Parameterisation: c[i] = S''(x_i).  On [x_i, x_{i+1}], t = x - x_i:
 *   S(x) = y[i] + b[i]*t + (c[i]/2)*t^2 + d[i]*t^3
 * where
 *   d[i] = (c[i+1] - c[i]) / (6*h[i])
 *   b[i] = (y[i+1] - y[i])/h[i] - h[i]*(2*c[i] + c[i+1])/6
 *
 * c[i] satisfies (i = 1..n-2, BCs c[0]=c[n-1]=0):
 *   h[i-1]*c[i-1] + 2*(h[i-1]+h[i])*c[i] + h[i]*c[i+1]
 *       = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
 *
 * Thomas algorithm: interior system is (n-2)×(n-2).
 *   plan->diag[k] = factored diagonal for interior row k (k=0..n-3).
 *
 * Thread safety: no global/static state.
 * ====================================================================== */
static spline_plan *spline_prepare(const double *restrict data_x, long data_size,
                                   int *err_out)
{
    if (data_size < 3) { *err_out = SPLINE_ERR_DATA_SIZE; return nullptr; }

    const long n  = data_size;
    const long nm = n - 1;   /* intervals       */
    const long ni = n - 2;   /* interior points */

    const size_t plan_sz = align_double(sizeof(spline_plan));
    const size_t h_sz    = (size_t)nm * sizeof(double);
    const size_t diag_sz = (size_t)ni * sizeof(double);
    const size_t total   = plan_sz + 2u*h_sz + diag_sz;

    unsigned char *mem = (unsigned char *)malloc(total);
    if (SPLINE_UNLIKELY(!mem)) { *err_out = SPLINE_ERR_OOM; return nullptr; }

    spline_plan *plan = reinterpret_cast<spline_plan *>(mem);
    plan->mem    = mem;
    plan->data_x = data_x;
    plan->n      = n;

    unsigned char *ptr = mem + plan_sz;
    plan->h     = reinterpret_cast<double *>(ptr); ptr += h_sz;
    plan->inv_h = reinterpret_cast<double *>(ptr); ptr += h_sz;
    plan->diag  = reinterpret_cast<double *>(ptr);

    double *restrict h     = plan->h;
    double *restrict inv_h = plan->inv_h;
    double *restrict diag  = plan->diag;

    /* Interval widths and reciprocals */
#if defined(__clang__)
#  pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#  pragma GCC ivdep
#endif
    for (long i = 0; i < nm; ++i) {
        const double hi = data_x[i+1] - data_x[i];
        if (SPLINE_UNLIKELY(hi == 0.0)) goto singular;
        h[i]     = hi;
        inv_h[i] = 1.0 / hi;
    }

    /*
     * Factor the (n-2)×(n-2) tridiagonal.
     *
     * Row k (k=0..ni-1) corresponds to interior point i=k+1.
     * diagonal[k] = 2*(h[k]+h[k+1])
     * sub-diag[k] (coeff of c[i-1] in row i) = h[k]
     * super-diag[k]                           = h[k+1]
     *
     * k=0: diag[0] = 2*(h[0]+h[1])  (unmodified)
     * k>=1:
     *   factor  = h[k] / diag[k-1]
     *   diag[k] = 2*(h[k]+h[k+1]) - factor*h[k]
     */
    diag[0] = 2.0 * (h[0] + h[1]);

    for (long k = 1; k < ni; ++k) {
        if (SPLINE_UNLIKELY(diag[k-1] == 0.0)) goto singular;
        const double factor = h[k] / diag[k-1];
        diag[k] = 2.0*(h[k] + h[k+1]) - factor*h[k];
    }
    if (SPLINE_UNLIKELY(diag[ni-1] == 0.0)) goto singular;

    return plan;

singular:
    free(mem);
    *err_out = SPLINE_ERR_SINGULAR;
    return nullptr;
}

static void spline_plan_free(spline_plan *plan) {
    if (plan) free(plan->mem);
}

/* =========================================================================
 * spline_interp_multi_tmpl<T>
 *
 * Core interpolation engine, templated on the value type T.
 *
 *   T = double               → real dataset (same maths as the original C)
 *   T = std::complex<double> → complex128 dataset, where data_y[d] points
 *                              to interleaved (re,im) pairs cast to T*.
 *                              Safe by C++11 §26.4: std::complex<double>
 *                              has guaranteed [re, im] memory layout.
 *
 * For T = double all spline_fma calls resolve to the non-template overload
 * which calls std::fma(), preserving hardware FMA on the real path.
 * For T = std::complex<double> the template overloads compute a*b+c.
 *
 * Scratch layout  (one std::vector<unsigned char> allocation):
 *   [  c_all : n * num_datasets * sizeof(T)  ]
 *   [  rhs   : ni * sizeof(T)                ]   (reused per dataset)
 *   [  idx_buf: out_size * sizeof(int)        ]
 *
 * std::bad_alloc on OOM is caught in the extern "C" wrappers.
 * ====================================================================== */
template<typename T>
static int spline_interp_multi_tmpl(
    const spline_plan *restrict plan,
    const T *const *restrict data_y,
    const double *restrict out_x,
    T *const *restrict out_y,
    long out_size, long num_datasets)
{
    if (out_size <= 0)    return SPLINE_ERR_OUT_SIZE;
    if (num_datasets <= 0) return SPLINE_ERR_NUM_DS;

    const long   n        = plan->n;
    const long   ni       = n - 2;
    const double *restrict data_x   = plan->data_x;
    const double *restrict h        = plan->h;
    const double *restrict inv_h    = plan->inv_h;
    const double *restrict diag_fac = plan->diag;

    /* Single scratch allocation — std::bad_alloc propagates to extern "C" */
    const size_t c_sz   = (size_t)n  * (size_t)num_datasets * sizeof(T);
    const size_t rhs_sz = (size_t)ni * sizeof(T);
    const size_t idx_sz = (size_t)out_size * sizeof(int);

    std::vector<unsigned char> scratch(c_sz + rhs_sz + idx_sz);

    T   *restrict c_all   = reinterpret_cast<T *>(scratch.data());
    T   *restrict rhs     = reinterpret_cast<T *>(scratch.data() + c_sz);
    int *restrict idx_buf = reinterpret_cast<int *>(scratch.data() + c_sz + rhs_sz);

    /* -------------------------------------------------------------------
     * Phase 1: Tridiagonal solve for c[1..n-2] (one per dataset).
     *
     * rhs[k] corresponds to interior point i = k+1.
     * raw_rhs[k] = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
     *            = 6*(y[i+1]*inv_h[i] - y[i]*(inv_h[i]+inv_h[i-1])
     *                 + y[i-1]*inv_h[i-1])
     *
     * Forward sweep:
     *   rhs[0] = raw_rhs[0]
     *   rhs[k] = raw_rhs[k] - (h[k]/diag[k-1])*rhs[k-1]  (k >= 1)
     *
     * Back-substitution:
     *   c[n-2] = rhs[ni-1] / diag[ni-1]
     *   c[k+1] = (rhs[k] - h[k+1]*c[k+2]) / diag[k]      (k = ni-2..0)
     * ------------------------------------------------------------------- */
    for (long d = 0; d < num_datasets; ++d) {
        const T *restrict y = data_y[d];
        T *restrict       c = c_all + d * n;

        c[0]     = T(0.0);
        c[n - 1] = T(0.0);

        /* k=0, i=1: no elimination needed */
        rhs[0] = T(6.0) * spline_fma(y[2], inv_h[1],
                              spline_fma(-y[1], inv_h[1] + inv_h[0],
                                          y[0] * inv_h[0]));

        /* k=1..ni-1, i=2..n-2 */
        for (long k = 1; k < ni; ++k) {
            const long i   = k + 1;
            const T    raw = T(6.0) * spline_fma(y[i+1], inv_h[i],
                                         spline_fma(-y[i], inv_h[i] + inv_h[i-1],
                                                     y[i-1] * inv_h[i-1]));
            const double factor = h[k] / diag_fac[k-1];
            rhs[k] = raw - factor * rhs[k-1];
        }

        /* Back-substitution */
        c[n-2] = rhs[ni-1] / diag_fac[ni-1];

        for (long k = ni-2; k >= 0; --k) {
            const long i = k + 1;
            c[i] = (rhs[k] - h[k+1] * c[i+1]) / diag_fac[k];
        }
    }

    /* -------------------------------------------------------------------
     * Phase 2a: Precompute interval indices for all output points.
     *
     * scratch owns this memory — SPLINE_ERR_BOUNDS return is leak-free.
     * ------------------------------------------------------------------- */
    {
        int prev_idx = 0;
        for (long j = 0; j < out_size; ++j) {
            const int idx = hunt(data_x, (int)n, out_x[j], prev_idx);
            if (SPLINE_UNLIKELY(idx < 0)) return SPLINE_ERR_BOUNDS;
            idx_buf[j] = idx;
            prev_idx = idx;
        }
    }

    /* -------------------------------------------------------------------
     * Phase 2b: Evaluate spline — d-outer for cache locality on c/y rows,
     *           j-inner for stride-1 writes to out_row.
     *
     * Horner form: S = y[i] + t*(b + t*(c[i]/2 + t*d))
     *
     * spline_fma dispatches to std::fma (HW FMA) when T=double, and to
     * a*b+c when T=std::complex<double>.
     * ------------------------------------------------------------------- */
    for (long d = 0; d < num_datasets; ++d) {
        const T *restrict y       = data_y[d];
        const T *restrict c       = c_all + d * n;
        T *restrict       out_row = out_y[d];

#if defined(__clang__)
#  pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#  pragma GCC ivdep
#endif
        for (long j = 0; j < out_size; ++j) {
            const int    idx     = idx_buf[j];
            const double t       = out_x[j] - data_x[idx];
            const double inv_hi  = inv_h[idx];
            const double hi_inv6 = h[idx] * (1.0 / 6.0);

            const T ci      = c[idx];
            const T ci1     = c[idx + 1];
            const T d_coeff = (ci1 - ci) * inv_hi * (1.0/6.0);
            const T b       = spline_fma(-hi_inv6,
                                         spline_fma(2.0, ci, ci1),
                                         (y[idx+1] - y[idx]) * inv_hi);

            out_row[j] = spline_fma(t,
                         spline_fma(t,
                         spline_fma(t, d_coeff, ci * 0.5),
                         b),
                         y[idx]);
        }
    }

    return SPLINE_OK;
}

/* =========================================================================
 * run_spline_multi<T> — RAII wrapper: prepare → eval → auto-free.
 *
 * Uses std::unique_ptr with spline_plan_free as a custom deleter so the
 * plan is freed on every return path, including exception propagation.
 * ====================================================================== */
template<typename T>
static int run_spline_multi(
    long data_size, long out_size, long num_datasets,
    const double *data_x, const T *const *data_y,
    const double *out_x, T *const *out_y)
{
    int err = SPLINE_OK;
    auto plan = std::unique_ptr<spline_plan, decltype(&spline_plan_free)>(
        spline_prepare(data_x, data_size, &err), spline_plan_free);
    if (!plan) return err;
    return spline_interp_multi_tmpl<T>(plan.get(), data_y, out_x, out_y,
                                       out_size, num_datasets);
}

/* =========================================================================
 * extern "C" public API
 *
 * These four functions preserve the exact ABI used by the Python ctypes
 * wrapper.  Each is a thin dispatcher that catches std::bad_alloc (which
 * std::vector raises on OOM) and maps it to SPLINE_ERR_OOM.
 * ====================================================================== */
extern "C" {

/* Single real dataset — convenience wrapper (num_datasets = 1). */
int spline_interp(const long data_size, const long out_size,
                  double const *const data_x, const double *const data_y,
                  double const *const out_x, double *out_y)
{
    if (out_size <= 0) return SPLINE_ERR_OUT_SIZE;
    try {
        const double *yd[1] = { data_y };
        double       *yo[1] = { out_y  };
        return run_spline_multi<double>(data_size, out_size, 1,
                                        data_x, yd, out_x, yo);
    } catch (const std::bad_alloc &) { return SPLINE_ERR_OOM; }
}

/* Multiple real datasets sharing one x-grid. */
int spline_interp_multi(const long data_size, const long out_size,
                        const long num_datasets,
                        const double *restrict data_x,
                        const double *const *restrict data_y,
                        const double *restrict out_x,
                        double *const *restrict out_y)
{
    if (out_size <= 0)    return SPLINE_ERR_OUT_SIZE;
    if (num_datasets <= 0) return SPLINE_ERR_NUM_DS;
    try {
        return run_spline_multi<double>(data_size, out_size, num_datasets,
                                        data_x, data_y, out_x, out_y);
    } catch (const std::bad_alloc &) { return SPLINE_ERR_OOM; }
}

/* Multiple complex128 datasets (interleaved double re,im pairs).
 *
 * data_y[d] points to data_size interleaved pairs: re0,im0,re1,im1,...
 * out_y[d]  points to out_size  interleaved pairs.
 *
 * Reinterpret-cast double** → std::complex<double>** is safe: C++11 §26.4
 * guarantees std::complex<double> has [re, im] layout == double[2].
 */
int spline_interp_multi_complex(const long data_size, const long out_size,
                                const long num_datasets,
                                const double *restrict data_x,
                                const double *const *restrict data_y,
                                const double *restrict out_x,
                                double *const *restrict out_y)
{
    if (out_size <= 0)    return SPLINE_ERR_OUT_SIZE;
    if (num_datasets <= 0) return SPLINE_ERR_NUM_DS;
    try {
        using CD = std::complex<double>;
        auto cy = reinterpret_cast<const CD *const *>(data_y);
        auto co = reinterpret_cast<CD *const *>(out_y);
        return run_spline_multi<CD>(data_size, out_size, num_datasets,
                                    data_x, cy, out_x, co);
    } catch (const std::bad_alloc &) { return SPLINE_ERR_OOM; }
}

} /* extern "C" */

static struct PyModuleDef _spline_interp_module = {
    PyModuleDef_HEAD_INIT,
    "_spline_interp",
    NULL,
    -1,
    NULL
};

PyMODINIT_FUNC PyInit__spline_interp(void) {
    return PyModule_Create(&_spline_interp_module);
}
