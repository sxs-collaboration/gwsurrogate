from distutils.core import setup
import sys

setup(name='gwsurrogate',
      version='0.4.0',
      author='Scott Field, Chad Galley',
      author_email='sfield@astro.cornell.edu',
      packages=['gwsurrogate'],
      license='MIT',
      contributors=[
      # Alphabetical by last name.
      "Jonathan Blackman"],
      description='An easy to use interface to gravitational wave surrogate models',
      # will start new downloads if these are installed in a non-standard location
      # install_requires=["numpy","matplotlib","scipy"],
      classifiers=[
                'Intended Audience :: Other Audience',
                'Intended Audience :: Science/Research',
                'Natural Language :: English',
                'License :: OSI Approved :: MIT License',
                'Programming Language :: Python',
                'Topic :: Scientific/Engineering',
                'Topic :: Scientific/Engineering :: Mathematics',
                'Topic :: Scientific/Engineering :: Physics',
      ],
)
