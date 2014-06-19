from distutils.core import setup
import sys

setup(name='gwsurrogate',
      version='0.2.1',
      author='Scott Field, Chad Galley',
      author_email='sfield@umd.edu',
      packages=['gwsurrogate'],
      license='MIT',
      description='an easy to use interface to gravitational wave surrogate models',
      install_requires=["numpy","matplotlib"],
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
