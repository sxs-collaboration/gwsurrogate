""" Gravitational Wave Surrogate classes for text and hdf5 files"""

from __future__ import division  # for py2

__copyright__ = "Copyright (C) 2014 Scott Field and Chad Galley"
__email__     = "sfield@astro.cornell.edu, crgalley@tapir.caltech.edu"
__status__    = "testing"
__author__    = "Kevin Barkett"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# adding "_" prefix to potentially unfamiliar module names
# so they won't show up in gws' tab completion
import numpy as np
from scipy.special import fresnel


################## NS Universal Relations

def UniversalRelationLambda2ToOmega2(lambda2):
  """ arXiv:1408.3789 eqn 3.5 with coeffs from Table I

   Input  -- ell=2 dimensionless tidal deformability lambda2 = 2/3*k2*C^5
   Output -- M_NS * omega2, the dimensionless frequency """

  if(lambda2<0):
    raise ValueError("ERROR: lambda2 value must be positive!")
  eta = np.log(lambda2)
  p = [0.182, -6.836e-3, -4.196e-3, 5.215e-4, -1.857e-5]
  omega2 = p[0] + eta*(p[1]+eta*(p[2]+eta*(p[3]+eta*(p[4]))))
  return omega2

def UniversalRelationLambda3ToOmega3(lambda3):
  """ arXiv:1408.3789 eqn 3.5 with coeffs from Table I

   Input  -- ell=3 dimensionless tidal deformability lambda3 = 2/15*k3*C^7
   Output -- M_NS * omega3, the dimensionless frequency """

  if(lambda3<0):
    raise ValueError("ERROR: lambda3 value must be positive!")
  eta = np.log(lambda3)
  p = [2.245e-1, -1.500e-2, -1.412e-3, 1.832e-4, -5.561e-6]
  omega3 = p[0] + eta*(p[1]+eta*(p[2]+eta*(p[3]+eta*(p[4]))))
  return omega3

def UniversalRelationLambda2ToLambda3(lambda2):
  """ arXiv:1311.0872 eqn 60 with coeffs from Table I

   Input  -- ell=2 dimensionless tidal deformability lambda2 = 2/3*k2*C^5
   Output -- ell=3 dimensionless tidal deformability lambda3 = 2/15*k3*C^7 """

  if(lambda2<0):
    raise ValueError("ERROR: lambda2 value must be positive!")
  eta = np.log(lambda2)
  p = [-1.15, 1.18, 2.51e-2, -1.31e-3, 2.52e-5]
  lambda3 = np.exp(p[0] + eta*(p[1]+eta*(p[2]+eta*(p[3]+eta*(p[4])))))
  return lambda3

def UniversalRelationLambda2ToAqm(lambda2):
  """ arXiv:1303.1528 eqn 54 with coeffs from Table I

   Input  -- ell=2 dimensionless tidal deformability lambda2 = 2/3*k2*C^5
   Output -- dimensionless rotationally-induced quadrupole moment (eqn 40)
             Aqm = -Q/(M^3 chi^2) """

  if(lambda2<0):
    raise ValueError("ERROR: lambda2 value must be positive!")
  eta = np.log(lambda2)
  p = [0.194, 0.0936, 0.0474, -4.21e-3, 1.23e-4]
  Aqm = np.exp(p[0] + eta*(p[1]+eta*(p[2]+eta*(p[3]+eta*(p[4])))))
  return Aqm

def UniversalRelationLambda2ToI(lambda2):
  """arXiv:1303.1528 eqn 54 with coeffs from Table I
   Input  -- ell=2 dimensionless tidal deformability lambda2 = 2/3*k2*C^5
   Output -- dimensionless moment of inertia (eqn 26); I = \bar{I}*M^3"""

  if(lambda2<0):
    raise ValueError("ERROR: lambda2 value must be positive!")
  eta = np.log(lambda2)
  p = [1.47, 0.0817, 0.0149, 2.87e-4, -3.64e-5]
  Aqm = np.exp(p[0] + eta*(p[1]+eta*(p[2]+eta*(p[3]+eta*(p[4])))))
  return Aqm

################## Tidal deformability functions

def EffectiveDeformabilityFromDynamicalTides(orbital_freq,omega_fmode,ell_mode,q):
  """arxiv:1608.01907, eqn 6.51

   Inputs:
     orbital_freq -- orbital frequency of the binary
     omega_fmode  -- f-mode dimensionless angular frequency of the ell-polar
                     mode of the deformed object
     ell_mode     -- ell-polar deformability (2 = quadrupolar; 3 = octopolar)
     q            -- mass ratio
   Outputs:
     The effective amplification of the ell-polar tidal deformability as a
     function of the oribital frequency """

  if(ell_mode not in [2,3]):
    raise ValueError("ERROR: 'ell_mode' only implemented for ell=2 or ell=3!")
  
  # If the input to the universal relations is too small (lambda2<1), then the
  #   fits can return unphysical negative resonance frequencies, in such cases
  #   just assume there is no significant resonance
  if(omega_fmode<=0):
    return np.ones(len(orbital_freq))

  # resonanace => freq_ratio == 1
  freq_ratio = omega_fmode / ell_mode / orbital_freq

  # Intermediate values
  # Because it only matters what the symmetric mass ratio is, XA*XB, it doesn't
  #   matter whether the input is q or 1/q
  X1 = q/(q+1.)
  X2 = 1. - X1
  epsilon = 256./5.*X1*X2*pow(omega_fmode/ell_mode,5./3.)
  t_hat = 8./5./np.sqrt(epsilon)*(1 - pow(freq_ratio,5./3.))
  omega_prime = 3./8.

  # Fresnel integral values
  ssa, csa = fresnel(t_hat*np.sqrt(2*omega_prime/np.pi))

  betaDTOverA = (
          pow(omega_fmode,2)/(pow(omega_fmode,2)-pow(ell_mode*orbital_freq,2))
          + pow(freq_ratio,2)/2./np.sqrt(epsilon)/t_hat/omega_prime
          + pow(freq_ratio,2)/np.sqrt(epsilon)*np.sqrt(np.pi/8./omega_prime)
          * (np.cos(omega_prime*t_hat*t_hat)*(1+2.*ssa)
            -np.sin(omega_prime*t_hat*t_hat)*(1+2.*csa))
          )
  # When 'orbital_freq' is close enough to 0, this formula seems to be
  #   numerically unstable though the asympotic limit should be 1. So when small
  #   enough, just set it to that value
  # NOTE: this cutoff value should match value used in
  #   EffectiveDissipativeDynamicalTides
  omega_cutoff = 5.e-5
  betaDTOverA[abs(orbital_freq)<=omega_cutoff] = 1.

  if ell_mode==2:
    return (1.+3.*betaDTOverA)/4.
  else:
    return (3.+5.*betaDTOverA)/8.

def EffectiveDissipativeDynamicalTides(orbitalFreq,effDefA,omegafModeA,XA):
  """ arXiv:1702.02053, eqn 15
   NOTE: there is an error in the paper that is correct in the XLALSim code!!
   This returns the effective correction to the ell=2 (quadrupole)
   deformability which used to compute the strain correction due to tidal
   effects """

  # NOTE: in the paper the input mass fraction is of the companion object, not
  # the object being deformed
  XB = 1.-XA
  effDissDyn = ((omegafModeA*omegafModeA*(effDefA-1.)
          + 6.*effDefA*XB*orbitalFreq*orbitalFreq)
        / (3.*orbitalFreq*orbitalFreq*(1.+2.*XB)))

  # When 'orbital_freq' is close enough to 0, this formula seems to be
  #   numerically unstable though the asympotic limit should be 1. So when small
  #   enough, just set it to that value
  # NOTE: this cutoff value should match value used in
  #   EffectiveDeformabilityFromDynamicalTides
  omega_cutoff = 5.e-5
  effDissDyn[abs(orbitalFreq)<=omega_cutoff] = 1.

  return effDissDyn


def Beta22_1(X):
  """ Coefficient for the 1PN tidal correction to h_22 mode
   Defined in arXiv:1203.4352, eqn A15
   Input
     X -- mass fraction of object being tidally deformed
   Output
     double, coeff for 1PN tidal correction to h_22 mode """ 

  return (-202.+560.*X-340.*X*X+45.*X*X*X)/42./(3.-2.*X)

def StrainTidalEnhancementFactor(lll,mmm,qqq,lambdaA,lambdaB,v):
  """ Strain enhancement by mode for the quadrupole (ell=2) love number
   Static correction is arXiv:1203.4352, eqn A14-A17
   Effective dynamical tide correction is in arXiv:1702.02053, eqn 15

   Because the strain correction is just an amplitude correction (specifically
     an amplification because the tidal effects enhance the GW radiation), just
     need to return the amplitude of the correction and ignore the phase of the
     mode

   If using effective tidal deformability from dynamic tides, must be included
     into lambdaA,lambdaB before passing them into this function """

  # mass fractions
  XA = qqq/(1.+qqq)
  XB = 1.-XA
  # Only beta22_1 is known, so set all others to 0, include them later into
  #   the formula when they are figured out
  beta22_1A = Beta22_1(XA)
  beta22_1B = Beta22_1(XB)
  #beta22_2A = beta22_2B = 0.
  #beta21_1A = beta21_1B = 0.
  #beta33_1A = beta33_1B = 0.
  #beta31_1A = beta31_1B = 0.

  # eta = symmetric mass ratio, MA*MB/M/M, XA*XB
  # delta = XA-XB
  eta = XA*XB

  v2 = v*v
  v4 = v2*v2
  v10 = v4*v4*v2

  # The leading part of the Newtonian expression, from arXiv:1310.1528, eqn 327
  # The formula used his is dimensionless with the M/R factor scaled out
  # (looks like there is a typo where the 'm' in the leading term is supposed
  # to be mass M?)
  newt = (8*eta*np.sqrt(np.pi/5))*v2
  newtlm = tidelm = 0

  # The kappaA_ell used in 1203.4352 is different from the dimensionless
  # lambdaA normally used, see eqn 8. In particular, we can see that (G=c=1)
  # kappaA_ell = lambdaA_ell * (2*ell-1)!! * XA^4 * (XB)
  kappaA_2 = lambdaA*3.*pow(XA,4)*(1.-XA)
  kappaB_2 = lambdaB*3.*pow(XB,4)*(1.-XB)

  if(lll==2 and mmm==2):
    newtlm = 1.
    tidelm = kappaA_2*(XA/XB+3.)*(1.+beta22_1A*v2) \
             + kappaB_2*(XB/XA+3.)*(1.+beta22_1B*v2)
  elif(lll==2 and mmm==1):
    newtlm = 1j/3.*v
    tidelm = - kappaA_2*(4.5-6.*XA) \
             + kappaB_2*(4.5-6.*XB)
  elif(lll==3 and mmm==3):
    newtlm = (-3./4.*1j*np.sqrt(15./14.))*v
    tidelm = - kappaA_2*6.*(1.-XA) \
             + kappaB_2*6.*(1.-XB)
  elif(lll==3 and mmm==1):
    newtlm = (1j/12./np.sqrt(14.))*v
    tidelm = - kappaA_2*6.*(1.-XA) \
             + kappaB_2*6.*(1.-XB)
  else:
    return 0.

  tidelm *= v10

  return abs(newt*newtlm*tidelm)

################## TaylorT2 PN Tidal Coefficients

## T2 Timing Terms ##

def PNT2QM_Tv4(XA,chiA):
  """ TaylorT2 2PN Quadrupole Moment Coefficient, v^4 Timing Term. 

   XA = mass fraction of object
   chiA = dimensionless spin of object """

  return -10.*XA*XA*chiA*chiA

def PNT2Tidal_Tv10(XA):
  """ TaylorT2 0PN Quadrupolar Tidal Coefficient, v^10 Timing Term. 
   XA = mass fraction of object """

  return 288-264*XA

def PNT2Tidal_Tv12(XA):
  """ TaylorT2 1PN Quadrupolar Tidal Coefficient, v^12 Timing Term. 
   XA = mass fraction of object """

  XATo2nd = XA*XA
  XATo3rd = XATo2nd*XA
  return (3179)/(4)-(919*XA)/(4)-(1143*XATo2nd)/(2) + 65*XATo3rd

def PNT2Tidal_Tv13(XA,chiA=0,chiB=0):
  """ TaylorT2 1.5PN Quadrupolar Tidal Coefficient, v^13 Timing Term. 

   XA = mass fraction of object
   chiA = aligned spin-orbit component of object
   chiB = aligned spin-orbit component of companion object """

  XATo2nd = XA*XA
  XATo3rd = XATo2nd*XA
  return -576*np.pi +(2496*np.pi*XA)/(5)+(324*XA+12*XATo2nd \
    - (1096*XATo3rd)/(5))*chiA+(588-(6616*XA)/(5)+ (4772*XATo2nd)/(5) \
    - (1096*XATo3rd)/(5))*chiB

def PNT2Tidal_Tv14(XA,chiA=0,chiB=0,AqmA=0,AqmB=0,alpha2PNT=0):
  """ TaylorT2 2PN Quadrupolar Tidal Coefficient, v^14 Timing Term. 

   XA = mass fraction of object
   chiA = aligned spin-orbit component of object
   chiB = aligned spin-orbit component of companion object
   AqmA = dimensionless spin-induced quadrupole moment of object
   AqmB = dimensionless spin-induced quadrupole moment of companion object
   alpha2PNT = 2PN Quadrupole Tidal Flux coefficient """

  XATo2nd = XA*XA
  XATo3rd = XATo2nd*XA
  XATo4th = XATo3rd*XA
  XATo5th = XATo4th*XA
  return (70312133/21168)+(4*alpha2PNT)/(3) - (147794303*XA)/(127008) \
    - (20905*XATo2nd)/(28) - (432193*XATo3rd)/(504)-(5848*XATo4th)/(9) \
    + (857*XATo5th)/(3) + (-(639*XATo2nd)/(2)+(525*XATo3rd)/(2) \
    + AqmA*(-312*XATo2nd+256*XATo3rd))*chiA*chiA \
    + (-609*XA+1108*XATo2nd-499*XATo3rd)*chiA*chiB \
    + (-(639)/(2)+(1803*XA)/(2)-(1689*XATo2nd)/(2) + (525*XATo3rd)/(2) \
    + AqmB*(-312+880*XA-824*XATo2nd+256*XATo3rd))*chiB*chiB

def PNT2Tidal_Tv15(XA,chiA=0,chiB=0):
  """ TaylorT2 2.5PN Quadrupolar Tidal Coefficient, v^15 Timing Term. 

   XA = mass fraction of object
   chiA = aligned spin-orbit component of object
   chiB = aligned spin-orbit component of companion object """

  XATo2nd = XA*XA
  XATo3rd = XATo2nd*XA
  XATo4th = XATo3rd*XA
  XATo5th = XATo4th*XA
  return -(241295*np.pi )/(98)+(216921*np.pi*XA)/(98)-(7528*np.pi*XATo2nd)/(7) \
    + (7142*np.pi*XATo3rd)/(7) + ((101949*XA)/(49)+(48875*XATo2nd)/(294) \
    - (78373*XATo3rd)/(147)-(3417*XATo4th)/(7) - (2574*XATo5th)/(7))*chiA \
    + ((637447)/(147)-(1026647*XA)/(98)+(931999*XATo2nd)/(98) \
    - (713938*XATo3rd)/(147)+(12977*XATo4th)/(7) - (2574*XATo5th)/(7))*chiB

def PNT2TidalOcto_Tv14(XA,beta0PNT=0):
  """ TaylorT2 0PN Octopolar Tidal Coefficient, v^14 Timing Term. 

   XA = mass fraction of object
   beta0PNT = 0PN Octopole Tidal Flux coefficient """

  return (4)/(3)*(520+beta0PNT)-(2080*XA)/(3)

def PNT2TidalOcto_Tv16(XA,beta0PNT=0,beta1PNT=0):
  """ TaylorT2 1PN Octopolar Tidal Coefficient, v^16 Timing Term. 

   XA = mass fraction of object
   beta0PNT = 0PN Octopole Tidal Flux coefficient
   beta1PNT = 1PN Octopole Tidal Flux coefficient """

  XATo2nd = XA*XA
  XATo3rd = XATo2nd*XA
  XATo4th = XATo3rd*XA
  XATo5th = XATo4th*XA
  return (33440)/(21)+(995*beta0PNT)/(168)+beta1PNT + ((24670)/(7) \
    + (17*beta0PNT)/(3))*XA-(17/6)*(1825+2*beta0PNT)*XATo2nd+(325*XATo3rd)/(6)

def PNT2TidalOcto_Tv17(XA,chiA=0,chiB=0,beta0PNT=0):
  """ TaylorT2 1.5PN Octopolar Tidal Coefficient, v^17 Timing Term. 

   XA = mass fraction of object
   chiA = aligned spin-orbit component of object
   chiB = aligned spin-orbit component of companion object
   beta0PNT = 0PN Octopole Tidal Flux coefficient """

  XATo2nd = XA*XA
  XATo3rd = XATo2nd*XA
  return -(64)/(9)*(260+beta0PNT)*np.pi + (16640*np.pi*XA)/(9) \
    + ((20)/(9)*(260+3*beta0PNT)*XA+(16)/(27)*(195+7*beta0PNT)*XATo2nd \
    - (2080*XATo3rd)/(3))*chiA + ((4)/(27)*(8580+73*beta0PNT) \
    - (4/27)*(21840+101*beta0PNT)*XA+(16)/(27)*(4485+7*beta0PNT)*XATo2nd \
    + (-2080*XATo3rd)/(3))*chiB

def PNT2TidalOcto_Tv18(XA,chiA=0,chiB=0,AqmA=0,AqmB=0,beta0PNT=0,beta1PNT=0, \
        beta2PNT=0):
  """ TaylorT2 2PN Octopolar Tidal Coefficient, v^18 Timing Term. 

   XA = mass fraction of object
   chiA = aligned spin-orbit component of object
   chiB = aligned spin-orbit component of companion object
   AqmA = dimensionless spin-induced quadrupole moment of object
   AqmB = dimensionless spin-induced quadrupole moment of companion object
   beta0PNT = 0PN Octopole Tidal Flux coefficient
   beta1PNT = 1PN Octopole Tidal Flux coefficient
   beta2PNT = 2PN Octopole Tidal Flux coefficient """

  XATo2nd = XA*XA
  XATo3rd = XATo2nd*XA
  XATo4th = XATo3rd*XA
  XATo5th = XATo4th*XA
  return (35111473)/(3969)+(6080015*beta0PNT)/(254016)+ (199*beta1PNT)/(42) \
    + (4*beta2PNT)/(5)+((68598877)/(7938)  + (16999*beta0PNT)/(840) \
    + (68*beta1PNT)/(15))*XA + ((7215185-16319*beta0PNT \
    - 11424*beta1PNT)*XATo2nd)/(2520) + (-(154907)/(8) \
    - (2477*beta0PNT)/(90))*XATo3rd+(-(3040/3)+(2477*beta0PNT)/(180))*XATo4th \
    + (455*XATo5th)/(18) + ((-858-(57*beta0PNT)/(10))*XATo2nd \
    + 858*XATo3rd+AqmA*((-832 - 28*beta0PNT/5)*XATo2nd \
    + 832*XATo3rd))*chiA*chiA \
    + ((-1612-11*beta0PNT)*XA+(3224 \
    + 11*beta0PNT)*XATo2nd+(-1612*XATo3rd))*chiA*chiB \
    + (-858-(57*beta0PNT)/(10)+(2574+57*beta0PNT/5)*XA \
    + (-2574-(57*beta0PNT)/(10))*XATo2nd + 858*XATo3rd \
    + AqmB*(-832-28/5*beta0PNT + (2496+56/5*beta0PNT)*XA \
    - (4/5)*(3120+7*beta0PNT)*XATo2nd + 832*XATo3rd))*chiB*chiB

def PNT2TidalOcto_Tv19(XA,chiA=0,chiB=0,beta0PNT=0,beta1PNT=0):
  """ TaylorT2 2.5PN Octopolar Tidal Coefficient, v^19 Timing Term. 

   XA = mass fraction of object
   chiA = aligned spin-orbit component of object
   chiB = aligned spin-orbit component of companion object
   beta0PNT = 0PN Octopole Tidal Flux coefficient
   beta1PNT = 1PN Octopole Tidal Flux coefficient """

  XATo2nd = XA*XA
  XATo3rd = XATo2nd*XA
  XATo4th = XATo3rd*XA
  XATo5th = XATo4th*XA
  return -(1)/(462)*(2604260+17705*beta0PNT+2688*beta1PNT)*np.pi \
    - 2/231*(516675 + beta0PNT)*np.pi*XA + (2)/(33)*(90640 \
    + 241*beta0PNT)*np.pi*XATo2nd + (152360*np.pi*XATo3rd)/(33) \
    + ((1)/(924)*(3242620+49273*beta0PNT+5040*beta1PNT)*XA \
    + ((4691970+82637*beta0PNT+4704*beta1PNT)*XATo2nd)/(1386) \
    + (1)/(693)*(-221335+91*beta0PNT)*XATo3rd \
    - (5)/(99)*(98807+254*beta0PNT)*XATo4th - (156910*XATo5th)/(99))*chiA \
    + ((23393100+277897*beta0PNT+24528*beta1PNT)/(2772) \
    + ((-44089360-337219*beta0PNT-33936*beta1PNT)*XA)/(2772) \
    + ((9394160-23497*beta0PNT+4704*beta1PNT)*XATo2nd)/(1386) \
    + ((-272870)/(231)+(563*beta0PNT)/(11))*XATo3rd \
    - (5)/(99)*(-68399+254*beta0PNT)*XATo4th-(156910*XATo5th)/(99))*chiB

## T2 Phasing Terms ##

def PNT2QM_Pv4(XA,chiA):
  """ TaylorT2 2PN Quadrupole Moment Coefficient, v^4 Phasing Term. 

   XA = mass fraction of object
   chiA = dimensionless spin of object """

  return -25.*XA*XA*chiA*chiA

def PNT2Tidal_Pv10(XA):
  """ TaylorT2 0PN Quadrupolar Tidal Coefficient, v^10 Phasing Term.

   XA = mass fraction of object """

  return 72-66*XA

def PNT2Tidal_Pv12(XA):
  """ TaylorT2 1PN Quadrupolar Tidal Coefficient, v^12 Phasing Term. 

   XA = mass fraction of object """

  XATo2nd = XA*XA
  XATo3rd = XATo2nd*XA
  return (15895)/(56)-(4595*XA)/(56) - (5715*XATo2nd)/(28)+(325*XATo3rd)/(14)

def PNT2Tidal_Pv13(XA,chiA=0,chiB=0):
  """ TaylorT2 1.5PN Quadrupolar Tidal Coefficient, v^13 Phasing Term. 

   XA = mass fraction of object
   chiA = aligned spin-orbit component of object
   chiB = aligned spin-orbit component of companion object """

  XATo2nd = XA*XA
  XATo3rd = XATo2nd*XA
  return -225*np.pi +195*np.pi*XA + ((2025*XA)/(16)+(75*XATo2nd)/(16) \
    - (685*XATo3rd)/(8))*chiA+((3675)/(16) - (4135*XA)/(8)+(5965*XATo2nd)/(16) \
    - (685*XATo3rd)/(8))*chiB

def PNT2Tidal_Pv14(XA,chiA=0,chiB=0,AqmA=0,AqmB=0,alpha2PNT=0):
  """ TaylorT2 2PN Quadrupolar Tidal Coefficient, v^14 Phasing Term. 

   XA = mass fraction of object
   chiA = aligned spin-orbit component of object
   chiB = aligned spin-orbit component of companion object
   AqmA = dimensionless spin-induced quadrupole moment of object
   AqmB = dimensionless spin-induced quadrupole moment of companion object
   alpha2PNT = 2PN Quadrupole Tidal Flux coefficient """

  XATo2nd = XA*XA
  XATo3rd = XATo2nd*XA
  XATo4th = XATo3rd*XA
  XATo5th = XATo4th*XA
  return (351560665)/(254016)+(5*alpha2PNT)/(9) - (738971515*XA)/(1524096) \
    - (104525*XATo2nd)/(336) - (2160965*XATo3rd)/(6048)-(7310*XATo4th)/(27) \
    + (4285*XATo5th)/(36) + (-(1065*XATo2nd)/(8)+(875*XATo3rd)/(8) \
    + AqmA*(-130*XATo2nd+(320*XATo3rd)/(3)))*chiA*chiA \
    + (-(1015*XA)/(4)+(1385*XATo2nd)/(3) - (2495*XATo3rd)/(12))*chiA*chiB \
    + (-(1065)/(8)+(3005*XA)/(8)-(2815*XATo2nd)/(8) + (875*XATo3rd)/(8) \
    + AqmB*(-130+(1100*XA)/(3)-(1030*XATo2nd)/(3)+(320*XATo3rd)/(3)))*chiB*chiB

def PNT2Tidal_Pv15(XA,chiA=0,chiB=0):
  """ TaylorT2 2.5PN Quadrupolar Tidal Coefficient, v^15 Phasing Term. 

   XA = mass fraction of object
   chiA = aligned spin-orbit component of object
   chiB = aligned spin-orbit component of companion object """

  XATo2nd = XA*XA
  XATo3rd = XATo2nd*XA
  XATo4th = XATo3rd*XA
  XATo5th = XATo4th*XA
  return -(241295*np.pi)/(224)+(216921*np.pi*XA)/(224)-(941*np.pi*XATo2nd)/(2) \
    + (3571*np.pi*XATo3rd)/(8) + ((101949*XA)/(112)+(48875*XATo2nd)/(672) \
    - (78373*XATo3rd)/(336)-(3417*XATo4th)/(16) - (1287*XATo5th)/(8))*chiA \
    + ((637447)/(336)-(1026647*XA)/(224)+(931999*XATo2nd)/(224) \
    - (356969*XATo3rd)/(168)+(12977*XATo4th)/(16) - (1287*XATo5th)/(8))*chiB

def PNT2TidalOcto_Pv14(XA,beta0PNT=0):
  """ TaylorT2 0PN Octopolar Tidal Coefficient, v^14 Phasing Term. 

   XA = mass fraction of object
   beta0PNT = 0PN Octopole Tidal Flux coefficient """

  return (5)/(9)*(520+beta0PNT)-(2600*XA)/(9)

def PNT2TidalOcto_Pv16(XA,beta0PNT=0,beta1PNT=0):
  """ TaylorT2 1PN Octopolar Tidal Coefficient, v^16 Phasing Term. 

   XA = mass fraction of object
   beta0PNT = 0PN Octopole Tidal Flux coefficient
   beta1PNT = 1PN Octopole Tidal Flux coefficient """

  XATo2nd = XA*XA
  XATo3rd = XATo2nd*XA
  XATo4th = XATo3rd*XA
  XATo5th = XATo4th*XA
  return (5*(267520+995*beta0PNT+168*beta1PNT))/(1848) \
    + (5/231)*(74010 + 119*beta0PNT)*XA \
    - (85)/(66)*(1825+2*beta0PNT)*XATo2nd + (1625*XATo3rd)/(66)

def PNT2TidalOcto_Pv17(XA,chiA=0,chiB=0,beta0PNT=0):
  """ TaylorT2 1.5PN Octopolar Tidal Coefficient, v^17 Phasing Term. 

   XA = mass fraction of object
   chiA = aligned spin-orbit component of object
   chiB = aligned spin-orbit component of companion object
   beta0PNT = 0PN Octopole Tidal Flux coefficient """

  XATo2nd = XA*XA
  XATo3rd = XATo2nd*XA
  return -(10)/(3)*(260+beta0PNT)*np.pi + (2600*np.pi*XA)/(3) \
    + ((25)/(24)*(260+3*beta0PNT)*XA+(5)/(18)*(195+7*beta0PNT)*XATo2nd \
    - 325*XATo3rd)*chiA + ((5)/(72)*(8580+73*beta0PNT) \
    - (5/72)*(21840 + 101*beta0PNT)*XA + (5)/(18)*(4485+7*beta0PNT)*XATo2nd \
    - 325*XATo3rd)*chiB

def PNT2TidalOcto_Pv18(XA,chiA=0,chiB=0,AqmA=0,AqmB=0,beta0PNT=0,beta1PNT=0, \
        beta2PNT=0):
  """ TaylorT2 2PN Octopolar Tidal Coefficient, v^18 Phasing Term. 

   XA = mass fraction of object
   chiA = aligned spin-orbit component of object
   chiB = aligned spin-orbit component of companion object
   AqmA = dimensionless spin-induced quadrupole moment of object
   AqmB = dimensionless spin-induced quadrupole moment of companion object
   beta0PNT = 0PN Octopole Tidal Flux coefficient
   beta1PNT = 1PN Octopole Tidal Flux coefficient
   beta2PNT = 2PN Octopole Tidal Flux coefficient """

  XATo2nd = XA*XA
  XATo3rd = XATo2nd*XA
  XATo4th = XATo3rd*XA
  XATo5th = XATo4th*XA
  return (25*(2247134272+6080015*beta0PNT+1203552*beta1PNT))/(13208832) \
    + (5*beta2PNT)/(13)+((1714971925)/(412776) + (84995*beta0PNT)/(8736) \
    + (85*beta1PNT)/(39))*XA - (5*(-7215185+16319*beta0PNT \
    + 11424*beta1PNT)*XATo2nd)/(26208) \
    - (5*(6970815+9908*beta0PNT)*XATo3rd)/(3744) \
    + (5*(-182400+2477*beta0PNT)*XATo4th)/(1872)+(875*XATo5th)/(72) \
    + (-(15/104)*(2860+19*beta0PNT)*XATo2nd + (825*XATo3rd)/(2) \
    + AqmA*((-400 - (35*beta0PNT)/(13))*XATo2nd+400*XATo3rd))*chiA*chiA \
    + ((-775-(275*beta0PNT)/(52))*XA+(1550 + (275*beta0PNT)/(52))*XATo2nd \
    - 775*XATo3rd)*chiA*chiB \
    + (-(15/104)*(2860+19*beta0PNT)+(15)/(52)*(4290 + 19*beta0PNT)*XA \
    - (15/104)*(8580+19*beta0PNT)*XATo2nd + (825*XATo3rd)/(2) \
    + AqmB*(-400-(35*beta0PNT)/(13)+(1200 + (70*beta0PNT)/(13))*XA \
    + (-1200-(35*beta0PNT)/(13))*XATo2nd + 400*XATo3rd))*chiB*chiB

def PNT2TidalOcto_Pv19(XA,chiA=0,chiB=0,beta0PNT=0,beta1PNT=0):
  """ TaylorT2 2.5PN Octopolar Tidal Coefficient, v^19 Phasing Term. 

   XA = mass fraction of object
   chiA = aligned spin-orbit component of object
   chiB = aligned spin-orbit component of companion object
   beta0PNT = 0PN Octopole Tidal Flux coefficient
   beta1PNT = 1PN Octopole Tidal Flux coefficient"""

  XATo2nd = XA*XA
  XATo3rd = XATo2nd*XA
  XATo4th = XATo3rd*XA
  XATo5th = XATo4th*XA
  return -(5*(2604260+17705*beta0PNT+2688*beta1PNT)*np.pi)/(4704) \
    - (5*(516675+1687*beta0PNT)*np.pi*XA)/(1176)+(5)/(168)*(90640 \
    + 241*beta0PNT)*np.pi*XATo2nd + (95225*np.pi*XATo3rd)/(42)  \
    + ((5*(3242620+49273*beta0PNT+5040*beta1PNT)*XA)/(9408) \
    + (5*(4691970+82637*beta0PNT+4704*beta1PNT)*XATo2nd)/(14112) \
    + (5*(-221335+91*beta0PNT)*XATo3rd)/(7056) \
    - (25*(98807+254*beta0PNT)*XATo4th)/(1008) - (392275*XATo5th)/(504))*chiA \
    + ((5*(23393100+277897*beta0PNT+24528*beta1PNT))/(28224) \
    - (5*(44089360+337219*beta0PNT+33936*beta1PNT)*XA)/(28224) \
    + (5*(9394160-23497*beta0PNT+4704*beta1PNT)/(14112))*XATo2nd \
    + (5*(-272870+11823*beta0PNT)*XATo3rd)/(2352) \
    - (25*(-68399+254*beta0PNT)*XATo4th)/(1008)-(392275*XATo5th)/(504))*chiB

################## TaylorT2 Tidal Evolution

def PNT2Tidal(v, q, lambda2A, lambda3A, AqmA, chizA, lambda2B, lambda3B, AqmB, \
        chizB, order=5):
  """ T2 Tidal corrections time and orbital phase of the binary's evolution.

   Inputs:
     v        -- PN expansion parameter, v = x^(1/2) = (M*Omega)^(1/3)
     q        -- mass ratio, defined as larger mass over smaller mass, MA/MB
     lambda2A -- ell=2 dimensionless tidal defomability of larger object
     lambda3A -- ell=3 dimensionless tidal defomability of larger object
     AqmA     -- dimensionless rotationally-induced quadrupole moment of
                 larger object
     chizA    -- spin component aligned with orbital angular momentum of larger
                 object
     lambda2B -- ell=2 dimensionless tidal defomability of smaller object
     lambda3B -- ell=3 dimensionless tidal defomability of smaller object
     AqmB     -- dimensionless rotationally-induced quadrupole moment of
                 smaller object
     chizB    -- spin component aligned with orbital angular momentum of
                 smaller object
     PNOrder  -- order of the PN tidal expansion beyond leading order, must be
                 one of (0,2,3,4,5)
  
   Outputs:
     dt_tid -- correction to the dimensionless time due to the tidal
               deformability as a function of the PN expansion parameter
     dp_tid -- correction to the dimensionless orbital phase due to the tidal
               deformability as a function of the PN expansion parameter
  
   NOTE: this code can utilize both static tides (the lambdas are input as
     a double) or dynamical tides (an array with length same as times array)"""

  # Sanity check the input
  # Either lambda2AB is an array for dynamical tides, or is a single double for
  #   static tides with a nonnegative value
  try:
    if(len(lambda2A)!=len(v)):
      raise IndexError("ERROR: lambda2A must be a float or be an array with "
              "the same length as the times array!")
  except TypeError:
    if(lambda2A<0):
      raise ValueError("ERROR: lambda2A inputs must be positive!")
  try:
    if(len(lambda2B)!=len(v)):
      raise IndexError("ERROR: lambda2B must be a float or be an array with "
              "the same length as the times array!")
  except TypeError:
    if(lambda2B<0):
      raise ValueError("ERROR: lambda2B inputs must be positive!")

  if(q<1):
    raise ValueError("ERROR: Mass ratio must be > 1 by definition used here!")

  if(abs(chizA)>=1 or abs(chizB)>=1):
    raise ValueError("ERROR: Spin must be < 1!")


  # alpha2PNT, beta?PNT are the missing tidal flux coefficients. Since these
  #   are likely to have a small affect on the waveform, default sets it to 0
  alpha2PNT = 0.
  beta0PNT  = 0.
  beta1PNT  = 0.
  beta2PNT  = 0.

  # Mass fractions
  delta = (q-1.)/(q+1.)
  XA = (delta+1.)/2.
  XB = 1.-XA

  if(order not in [0,2,3,4,5]):
    raise ValueError("ERROR: order must be one of [0,2,3,4,5]!")

  v2 = v*v
  v4 = v2*v2
  v5 = v4*v

  XATo4th = XA*XA*XA*XA
  XATo6th = XATo4th*XA*XA

  XBTo4th = XB*XB*XB*XB
  XBTo6th = XBTo4th*XB*XB

  # Quadrupole tide terms (start at 5PN)
  t_tid = lambda2A*XATo4th*PNT2Tidal_Tv10(XA) \
            + lambda2B*XBTo4th*PNT2Tidal_Tv10(XB)
  p_tid = lambda2A*XATo4th*PNT2Tidal_Pv10(XA) \
            + lambda2B*XBTo4th*PNT2Tidal_Pv10(XB)
  if (order>=2):
    t_tid += v2*(lambda2A*XATo4th*PNT2Tidal_Tv12(XA) \
              + lambda2B*XBTo4th*PNT2Tidal_Tv12(XB))
    p_tid += v2*(lambda2A*XATo4th*PNT2Tidal_Pv12(XA) \
              + lambda2B*XBTo4th*PNT2Tidal_Pv12(XB))
  if (order>=3):
    t_tid += v2*v*(lambda2A*XATo4th*PNT2Tidal_Tv13(XA,chizA,chizB) \
              + lambda2B*XBTo4th*PNT2Tidal_Tv13(XB,chizB,chizA))
    p_tid += v2*v*(lambda2A*XATo4th*PNT2Tidal_Pv13(XA,chizA,chizB) \
              + lambda2B*XBTo4th*PNT2Tidal_Pv13(XB,chizB,chizA))
  if (order>=4):
    t_tid += v4*(lambda2A*XATo4th*PNT2Tidal_Tv14(XA,chizA,chizB,AqmA,AqmB,alpha2PNT) \
              + lambda2B*XBTo4th*PNT2Tidal_Tv14(XB,chizB,chizA,AqmB,AqmA,alpha2PNT))
    p_tid += v4*(lambda2A*XATo4th*PNT2Tidal_Pv14(XA,chizA,chizB,AqmA,AqmB,alpha2PNT) \
              + lambda2B*XBTo4th*PNT2Tidal_Pv14(XB,chizB,chizA,AqmB,AqmA,alpha2PNT))
  if (order==5):
    t_tid += v5*(lambda2A*XATo4th*PNT2Tidal_Tv15(XA,chizA,chizB) \
              + lambda2B*XBTo4th*PNT2Tidal_Tv15(XB,chizB,chizA))
    p_tid += v5*(lambda2A*XATo4th*PNT2Tidal_Pv15(XA,chizA,chizB) \
              + lambda2B*XBTo4th*PNT2Tidal_Pv15(XB,chizB,chizA))

  # Octopole tide terms (start at 7PN)
  t_tid += v4*(lambda3A*XATo6th*PNT2TidalOcto_Tv14(XA,beta0PNT) \
            + lambda3B*XBTo6th*PNT2TidalOcto_Tv14(XB,beta0PNT))
  p_tid += v4*(lambda3A*XATo6th*PNT2TidalOcto_Pv14(XA,beta0PNT) \
            + lambda3B*XBTo6th*PNT2TidalOcto_Pv14(XB,beta0PNT))
  if (order>=2):
    t_tid += v5*v*(lambda3A*XATo6th*PNT2TidalOcto_Tv16(XA,beta0PNT,beta1PNT) \
                + lambda3B*XBTo6th*PNT2TidalOcto_Tv16(XB,beta0PNT,beta1PNT))
    p_tid += v5*v*(lambda3A*XATo6th*PNT2TidalOcto_Pv16(XA,beta0PNT,beta1PNT) \
                + lambda3B*XBTo6th*PNT2TidalOcto_Pv16(XB,beta0PNT,beta1PNT))
  if (order>=3):
    t_tid += v5*v2*(lambda3A*XATo6th*PNT2TidalOcto_Tv17(XA,chizA,chizB,beta0PNT) \
                + lambda3B*XBTo6th*PNT2TidalOcto_Tv17(XB,chizB,chizA,beta0PNT))
    p_tid += v5*v2*(lambda3A*XATo6th*PNT2TidalOcto_Pv17(XA,chizA,chizB,beta0PNT) \
                + lambda3B*XBTo6th*PNT2TidalOcto_Pv17(XB,chizB,chizA,beta0PNT))
  if (order>=4):
    t_tid += v4*v4*(lambda3A*XATo6th*PNT2TidalOcto_Tv18(XA,chizA,chizB,AqmA,AqmB,beta0PNT,beta1PNT,beta2PNT) \
                  + lambda3B*XBTo6th*PNT2TidalOcto_Tv18(XB,chizB,chizA,AqmB,AqmA,beta0PNT,beta1PNT,beta2PNT))
    p_tid += v4*v4*(lambda3A*XATo6th*PNT2TidalOcto_Pv18(XA,chizA,chizB,AqmA,AqmB,beta0PNT,beta1PNT,beta2PNT) \
                  + lambda3B*XBTo6th*PNT2TidalOcto_Pv18(XB,chizB,chizA,AqmB,AqmA,beta0PNT,beta1PNT,beta2PNT))
  if (order==5):
    t_tid += v5*v4*(lambda3A*XATo6th*PNT2TidalOcto_Tv19(XA,chizA,chizB,beta0PNT,beta1PNT) \
                + lambda3B*XBTo6th*PNT2TidalOcto_Tv19(XB,chizB,chizA,beta0PNT,beta1PNT))
    p_tid += v5*v4*(lambda3A*XATo6th*PNT2TidalOcto_Pv19(XA,chizA,chizB,beta0PNT,beta1PNT) \
                + lambda3B*XBTo6th*PNT2TidalOcto_Pv19(XB,chizB,chizA,beta0PNT,beta1PNT))

  dt_tid  = -5./(256.*XA*XB)*v2*t_tid
  dt_tid += -5./(256.*XA*XB)/v4*((AqmA-1)*PNT2QM_Tv4(XA,chizA)+(AqmB-1)*PNT2QM_Tv4(XB,chizB))
  dp_tid  = -1./(32.*XA*XB)*v5*p_tid
  dp_tid += -1./(32.*XA*XB)/v*((AqmA-1)*PNT2QM_Pv4(XA,chizA)+(AqmB-1)*PNT2QM_Pv4(XB,chizB))
  return  dt_tid, dp_tid


