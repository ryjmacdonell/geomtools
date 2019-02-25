"""
Setup script for the Gimbal package
"""
from setuptools import setup
from setuptools import find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


# read the current version number
exec(open('gimbal/_version.py').read())


setup(
    name='gimbal',
    version=__version__,
    description=('Tools for importing, creating, editing and querying ' +
                 'molecular geometries'),
    long_description=readme(),
    keywords='gimbal molecule geometry displacement transformation 3D',
    url='https://github.com/ryjmacdonell/gimbal',
    author='Ryan J. MacDonell',
    author_email='rmacd054@uottawa.ca',
    license='MIT',
    packages=find_packages(),
    scripts=['bin/convgeom'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Chemistry'
                 ],
    install_requires=['numpy>=1.6.0', 'scipy>=0.9.0']
      )
