from distutils.core import setup
import sys


# Encoding
major, minor1, minor2, release, serial = sys.version_info
if major >= 3:
    def rd(filename):
        f = open(filename, encoding="utf-8")
        r = f.read()
        f.close()
        return r
else:
    def rd(filename):
        f = open(filename)
        r = f.read()
        f.close()
        return r

setup(name='gwsurrogate',
      version='0.1',
      author='Scott Field, Chad Galley',
      author_email='sfield@umd.edu',
      packages=['gwsurrogate'],
      license='GPL',
      description='an easy to use interface to gravitational wave surrogate models',
      long_description="\n\n"+rd("README"),
      install_requires=["numpy","matplotlib","h5py"],
      classifiers=[
                'Intended Audience :: Other Audience',
                'Intended Audience :: Science/Research',
                'Natural Language :: English',
                'License :: OSI Approved :: GNU General Public License (GPL)',
                'Programming Language :: Python',
                'Topic :: Scientific/Engineering',
                'Topic :: Scientific/Engineering :: Mathematics',
                'Topic :: Scientific/Engineering :: Physics',
                ],

      )

