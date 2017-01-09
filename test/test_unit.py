"""
Simple unit tests.
"""

from __future__ import division
import nose
import numpy as np
import gwsurrogate as gws
import os

path_to_surrogate = 'tutorial/TutorialSurrogate/EOB_q1_2_NoSpin_Mode22/'

def test_orbital_symmetry_flags():
  # TODO: add 4d2s -- this will give non-trival combinations of this flag
  """ Check valid combinations of orbital symmetry flags"""

  EOBNRv2_sur = gws.EvaluateSurrogate(path_to_surrogate)

  # These three combinations should pass
  EOBNRv2_sur = gws.EvaluateSurrogate(path_to_surrogate, use_orbital_plane_symmetry=True)
  modes, t, hp, hc = EOBNRv2_sur(q=1.14,ell=[2],m=[2],mode_sum=False,fake_neg_modes=True)

  EOBNRv2_sur = gws.EvaluateSurrogate(path_to_surrogate, use_orbital_plane_symmetry=True)
  modes, t, hp, hc = EOBNRv2_sur(q=1.14,ell=[2],m=[2],mode_sum=False,fake_neg_modes=False)

  EOBNRv2_sur = gws.EvaluateSurrogate(path_to_surrogate, use_orbital_plane_symmetry=False)
  modes, t, hp, hc = EOBNRv2_sur(q=1.14,ell=[2],m=[2],mode_sum=False,fake_neg_modes=False)

  # This should fail with a ValueError. So seeing this error is a passed test
  try:
    EOBNRv2_sur = gws.EvaluateSurrogate(path_to_surrogate, use_orbital_plane_symmetry=False)
    modes, t, hp, hc = EOBNRv2_sur(q=1.14,ell=[2],m=[2],mode_sum=False,fake_neg_modes=True)
  except ValueError:
    pass
