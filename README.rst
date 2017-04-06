GeomTools
=========
Tools in Python for importing, creating, editing and querying molecular
geometries.

Usage
-----
GeomTools is intended for usage with geometry file formats used by
electronic structure programs and molecular modelling software. For example,
an XYZ file for the nitrogen molecule (n2.xyz) may contain::

    2

    N   0.000   0.000  -0.550
    N   0.000   0.000   0.550

Opening an input file is as simple as::

    >>> from geomtools import molecule
    >>> my_mol = molecule.import_molecule('n2.xyz')

Querying and changing the molecular structure can then be done from the
Molecule object::

    >>> my_mol.get_stre([1, 2])
    1.100
    >>> my_mol.translate(-1.0, [0, 0, 1], ind=[1]) # translate atom 1 by -1.0 Angstrom along the z-axis
    >>> my_mol.get_stre([1, 2])
    2.100

Note that the indices of atoms in the molecule start from 1. The index 0 is
reserved for the centre of mass of the Molecule object. The import_bundle
function can be used to import multiple geometries from a single file to
a MoleculeBundle object.

Other modules contain more advanced features such as finding bonds and
internal coordinates (bonding), displacing and aligning structures (displace),
reading, writing and converting geometry files (fileio) and mapping structures
onto reference geometries (kabsch).

Formats
-------
Currently supports geometry files in XYZ, COLUMBUS and Z-matrix formats
as well as custom formats (Geometry.dat and FMS TrajDump) for import or
export. The format is automatically detected during import if no format
is given. Output formats are assigned by the filename extension. Formatting
details are given in fileio.py.

Installation
------------
To add to your local Python packages, clone the repository and use setup.py
to install via::

    $ git clone https://github.com/ryjmacdonell/geomtools.git
    $ python setup.py install

This will also add a script 'convgeom' to the path which can be used to
quickly convert files between formats.

Requirements
------------
Requires at least Python 3.3, NumPy v1.6.0 and SciPy v0.9.0
