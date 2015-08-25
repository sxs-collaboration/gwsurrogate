import sys
if hasattr(sys, '_called_from_test'):
  print "called from within a test run"
  # older constants (outdated 7/9/2015)
  # KEEP ME!
  c         = 2.99792458e8     # Speed of light (MKS)
  G         = 6.67428e-11      # Gravitation constant (MKS)
  Msun      = 1.98892e30       # Solar mass (kg)
  Msuninsec = Msun * G / c**3  # Solar mass (secs)
  Mpcinm    = 3.08568025e22    # Megaparsecs (m)
  Mpcinkm   = Mpcinm/1000.0    # Megaparsecs (km)

else:
  print "gwsurrogate: using LAL values for constants"
  # These contants are found in LAL code
  c         = 299792458.0
  G         = 6.67384e-11
  Msun      = 1.9885469549614615e+30
  Msuninsec = Msun * G / c**3
  Mpcinm    = 3.085677581491367e+22
  Mpcinkm   = Mpcinm/1000.0    # Megaparsecs (km)

# Jonathan's constants
#Msun = 1.9891e30
#Mpcinm = 3.08567758e22
#SPEED_OF_LIGHT_C = 299792458
#GRAVITATIONAL_CONSTANT_G = 6.67384e-11
#G = 6.67384e-11
#Msuninsec = Msun * G / c**3
#Mpcinkm   = Mpcinm/1000.0    # Megaparsecs (km)

