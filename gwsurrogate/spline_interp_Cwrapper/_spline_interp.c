#include <stdio.h>
#include <gsl/gsl_spline.h>

void spline_interp(long data_size, long out_size, \
        double *data_x, double *data_y, \
        double *out_x, double *out_y) {

    // initialize
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline    = gsl_spline_alloc(gsl_interp_cspline, data_size);
    gsl_spline_init(spline, data_x, data_y, data_size);

    // evaluate
    int ii;
    for (ii=0; ii < out_size; ii++) {
        out_y[ii] = gsl_spline_eval (spline, out_x[ii], acc);
    }

    // free memory
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
}
