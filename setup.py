"""
Setup script for the Gimbal package
"""
from setuptools import setup
from setuptools import find_packages


def readme():
    """Returns the contents of the README without the header image."""
    header = '======\nGimbal\n======\n'
    with open('README.rst', 'r') as f:
        f.readline()
        return header + f.read()


def requirements():
    """Returns the requirement list."""
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f.readlines()]


# read the current version number
exec(open('gimbal/_version.py').read())


setup(
    name='gimbal',
    version=__version__,
    description=('Tools for importing, creating, editing and querying ' +
                 'molecular geometries'),
    long_description=readme(),
    long_description_content_type='text/x-rst',
    keywords='gimbal molecule geometry displacement transformation 3D',
    url='https://github.com/ryjmacdonell/gimbal',
    author='Ryan J. MacDonell',
    author_email='rmacd054@uottawa.ca',
    license='MIT',
    packages=find_packages(),
    scripts=['bin/convgeom', 'bin/measure', 'bin/nudge', 'bin/subst'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Chemistry'
                 ],
    install_requires=requirements()
      )
