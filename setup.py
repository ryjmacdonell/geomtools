"""
Setup script for the geomtools package
"""
from setuptools import setup
from setuptools import find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='geomtools',
    version='0.1',
    description=('Tools for importing, creating, editing and querying ' +
                 'molecular geometries'),
    long_description=readme(),
    keywords='geomtools molecules geometry displacement',
    url='https://github.com/ryjmacdonell/geomtools',
    author='Ryan J. MacDonell',
    author_email='rmacd054@uottawa.ca',
    license='MIT',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Chemistry'
                 ],
    install_requires=['numpy>=1.6.0', 'scipy>=0.9.0']
      )
