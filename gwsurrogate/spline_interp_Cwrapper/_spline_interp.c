#include <stdio.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

int spline_interp(const long data_size, const long out_size,
        const double *data_x, const double *data_y,
        const double *out_x, double *out_y) {

    int status = 0;

    // initialize
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline    = gsl_spline_alloc(gsl_interp_cspline, data_size);
    gsl_spline_init(spline, data_x, data_y, data_size);

    // save original error handler; check error status below
    gsl_error_handler_t *old_handler = gsl_set_error_handler_off();

    // evaluate
    int ii;
    for (ii=0; ii < out_size; ii++) {
      status = gsl_spline_eval_e (spline, out_x[ii], acc, out_y+ii);
      if ( status ) {
        // error, bail out
        break;
      }
    }

    // restore original error handler
    gsl_set_error_handler(old_handler);

    // free memory
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);

    // 0 for success
    return status;
}
