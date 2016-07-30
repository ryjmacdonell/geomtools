"""
The Molecule object and tools for generating and querying molecular geometries.

Creates a saved copy of the geometry after input for reversion after an
operation. Can add/remove individual atoms or groups or set the full geometry.
"""
import numpy as np
import geomtools.fileio as fileio
import geomtools.displace as displace


class Molecule(object):
    """
    Object containing the molecular geometry and functions for setting and
    getting geometric properties.
    """
    def __init__(self, elem=np.array([], dtype=str), xyz=np.empty((0, 3)),
                 comment=''):
        self.elem = elem
        self.xyz = xyz
        self.comment = comment
        self.natm = len(elem)
        self.saved = True
        self.save()

    def _check(self):
        """Checks that len(elem) = len(xyz)."""
        len_elem = len(self.elem)
        len_xyz = len(self.xyz)
        elif len_elem != len_xyz:
            raise ValueError('Number of element labels ({:d}) not equal '
                             'to number of cartesian vectors '
                             '({:d}).'.format(len_elem, len_xyz))

    def copy(self, comment=None):
        """Creates a copy of the Molecule object."""
        self._check()
        if comment is None:
            comment = 'Copy of ' + self.comment
        return Molecule(np.copy(self.elem), np.copy(self.xyz), comment)

    def save(self):
        """Saves molecular properties to 'orig' variables."""
        self._check()
        self.orig_elem = np.copy(self.elem)
        self.orig_xyz = np.copy(self.xyz)
        self.saved = True

    def revert(self):
        """Reverts properties to 'orig' variables."""
        if not self.saved:
            self.elem = np.copy(self.orig_elem)
            self.xyz = np.copy(self.orig_xyz)
        self.saved = True

    def set_geom(self, elem, xyz):
        """Sets molecular geometry."""
        self.elem = elem
        self.xyz = xyz
        self._check()
        self.saved = False

    def add_comment(self, comment):
        """Adds a comment line to describe the molecule."""
        self.comment = comment

    def add_atoms(self, new_elem, new_xyz):
        """Adds atoms(s) to molecule."""
        self.natm += 1 if isinstance(new_elem, str) else len(new_elem)
        self.elem = np.hstack((self.elem, new_elem))
        self.xyz = np.vstack((self.xyz, new_xyz))
        self._check()
        self.saved = False

    def rm_atoms(self, ind):
        """Removes atom(s) from molecule by index."""
        self.natm -= 1 if isinstance(ind, int) else len(ind)
        self.elem = np.delete(self.elem, ind)
        self.xyz = np.delete(self.xyz, ind, axis=0)
        self._check()
        self.saved = False

    def rearrange(self, new_ind, old_ind=None):
        """Moves atom(s) from old_ind to new_ind."""
        _rearrange_check(new_ind, old_ind, self.natm)
        self.xyz[old_ind] = self.xyz[new_ind]

    # Input/Output
    def read(self, infile, fmt='xyz', hc=False):
        """Reads single geometry from input file in provided format."""
        read_func = getattr(fileio, 'read_' + fmt)
        self.elem, self.xyz, self.comment = read_func(infile, hascomment=hc)
        self.save()

    def write(self, outfile, fmt='xyz'):
        """Writes geometry to an output file in provided format."""
        write_func = getattr(fileio, 'write_' + fmt)
        write_func(outfile, self.elem, self.xyz, self.comment)

    # Accessors
    def get_natm(self):
        """Returns number of atoms."""
        return self.natm

    def get_elem(self):
        """Returns list of elements."""
        return self.elem

    def get_xyz(self):
        """Returns cartesian geometry."""
        return self.xyz

    def get_comment(self):
        """Returns comment line."""
        return self.comment

    # Internal geometry
    def get_stre(self, ind, units='ang'):
        """Returns bond length based on index in molecule."""
        return displace.stre(self.xyz, ind, units=units)

    def get_bend(self, ind, units='rad'):
        """Returns bond angle based on index in molecule."""
        return displace.bend(self.xyz, ind, units=units)

    def get_tors(self, ind, units='rad'):
        """Returns dihedral angle based on index in molecule."""
        return displace.tors(self.xyz, ind, units=units)

    def get_oop(self, ind, units='rad'):
        """Returns out-of-plane angle based on index in molecule."""
        return displace.oop(self.xyz, ind, units=units)


class MoleculeBundle(object):
    """
    Object containing a set of molecules in the form of Molecule
    objects.
    """
    def __init__(self, molecules=None):
        if molecules is None:
            self.molecules = np.array([], dtype=object)
        else:
            self.molecules = np.array(molecules, dtype=object)
        self.nmol = len(self.molecules)
        self._check()

    def _check(self):
        """Check that all bundle objects are Molecule type."""
        for mol in self.molecules:
            if not isinstance(mol, Molecule):
                raise TypeError('Elements of molecule bundle must be '
                                'Molecule type.')

    def copy(self):
        """Returns a copy of the MoleculeBundle object."""
        return MoleculeBundle([mol.copy() for mol in self.molecules])

    def save(self):
        """Saves all molecules in the bundle."""
        for mol in self.molecules:
            mol.save()
        self._check()

    def revert(self):
        """Reverts each molecule in the bundle."""
        for mol in self.molecules:
            mol.revert()

    def rearrange(self, new_ind, old_ind=None):
        """Moves molecule(s) from old_ind to new_ind in bundle."""
        _rearrange_check(new_ind, old_ind, self.nmol)
        self.molecules[old_ind] = self.molecules[new_ind]

    def add_molecules(self, new_molecules):
        """Adds molecule(s) to the bundle."""
        self.nmol += (1 if isinstance(new_molecules, Molecule)
                      else len(molecules))
        self.molecules = np.hstack((self.molecules, molecules))
        self._check()

    def rm_molecules(self, ind):
        """Removes molecule(s) from the bundle by index."""
        self.nmol -= 1 if isinstance(ind, int) else len(ind)
        self.molecules = np.delete(self.molecules, ind)

    # Input/Output
    def read(self, infile, fmt='xyz', hc=False):
        """Reads all geometries from input file in provided format."""
        read_func = getattr(fileio, 'read_' + fmt)
        while True:
            try:
                new_mol = Molecule(*read_func(infile, hascomment=hc))
                self.molecules = np.hstack((self.molecules, new_mol))
                self.nmol += 1
            except ValueError:
                break

        self.save()

    def write(self, outfile, fmt='xyz'):
        """Writes geometries to an output file in provided format."""
        for mol in self.molecules:
            mol.write(outfile, fmt=fmt)

    # Accessors
    def get_nmol(self):
        """Returns the number of molecules."""
        return self.nmol

    def get_molecules(self):
        """Returns the list of molecules."""
        return self.molecules


def _rearrange_check(new_ind, old_ind, ntot):
    """Checks indices of rearrangement routines for errors."""
    if old_ind is None:
        old_ind = range(ntot)
    new = [new_ind] if isinstance(new_ind, int) else new_ind
    old = [old_ind] if isinstance(old_ind, int) else old_ind
    if len(new) != len(old):
        raise IndexError('Old and new indices must be the same length')


def import_molecule(fname, fmt='xyz', hc=False):
    """Imports geometry in provided format to Molecule object."""
    read_func = getattr(fileio, 'read_' + fmt)
    with open(fname, 'r') as infile:
        return Molecule(*read_func(infile, hascomment=hc))


def import_bundle(fname, fmt='xyz', hc=False):
    """Imports geometries in provided format to MoleculeBundle object."""
    read_func = getattr(fileio, 'read_' + fmt)
    molecules = []
    with open(fname, 'r') as infile:
        while True:
            try:
                molecules.append(Molecule(*read_func(infile, hascomment=hc)))
            except ValueError:
                break

    return MoleculeBundle(molecules)
