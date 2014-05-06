# --- test2.py ---

""" Testing EvaluateSurrogate class in surrogate.py """

import numpy as np
import surrogate


readpath = 'Surrogate_l2_m2_test.h5'

sur = surrogate.EvaluateSurrogate(readpath)