Getting started
===============

Many of the Gimbal features are accessible through the Molecule and
MoleculeBundle objects.


Importing molecules
-------------------

Given an input geometry file such as `n2.xyz`::

     2

    N    0.000000    0.000000    0.564990
    N    0.000000    0.000000   -0.564990

the file can be read to a Molecule object using::

    >>> import gimbal as gb
    >>> n2 = gb.import_molecule('n2.xyz')

Files in XYZ, COLUMBUS, MOLPRO, Z-Matrix and FMS trajectory formats can
be read automatically.
