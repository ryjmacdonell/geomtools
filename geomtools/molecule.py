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
    def __init__(self, natm=0, elem=np.array([], dtype=str),
                 xyz=np.empty((0, 3)), comment=''):
        self.natm = natm
        self.elem = elem
        self.xyz = xyz
        self.comment = comment
        self.saved = True
        self.save()

    def _check(self):
        """Checks that natm = len(elem) = len(xyz)."""
        natm = self.natm
        len_elem = len(self.elem)
        len_xyz = len(self.xyz)
        if natm != len_elem and natm != len_xyz:
            raise ValueError('Number of atoms ({:d}) not equal to number of '
                             'element labels ({:d}) and number of cartesian '
                             'vectors ({:d}).'.format(natm, len_elem, len_xyz))
        elif natm != len_elem:
            raise ValueError('Number of element labels ({:d}) not equal to '
                             'number of atoms ({:d}).'.format(natm, len_elem))
        elif natm != len_xyz:
            raise ValueError('Number of cartesian vectors ({:d}) not equal to '
                             'number of atoms ({:d}).'.format(natm, len_xyz))

    def copy(self):
        """Creates a copy of the Molecule object."""
        self._check()
        return Molecule(np.copy(self.natm), np.copy(self.elem),
                        np.copy(self.xyz))

    def save(self):
        """Saves molecular properties to 'orig' variables."""
        self._check()
        self.orig_natm = np.copy(self.natm)
        self.orig_elem = np.copy(self.elem)
        self.orig_xyz = np.copy(self.xyz)
        self.saved = True

    def revert(self):
        """Reverts properties to 'orig' variables."""
        if not self.saved:
            self.natm = np.copy(self.orig_natm)
            self.elem = np.copy(self.orig_elem)
            self.xyz = np.copy(self.orig_xyz)
        self.saved = True

    def set_geom(self, natm, elem, xyz):
        """Sets molecular geometry."""
        self.natm = natm
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

    def read(self, infile, fmt='xyz', hc=False):
        """Reads single geometry from input file in provided format."""
        read_func = getattr(fileio, 'read_' + fmt)
        (self.natm, self.elem, self.xyz,
         self.comment) = read_func(infile, hascomment=hc)
        self.save()

    def write(self, outfile, fmt='xyz'):
        """Writes geometry to an output file in provided format."""
        write_func = getattr(fileio, 'write_' + fmt)
        write_func(outfile, self.natm, self.elem, self.xyz, self.comment)

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
    def __init__(self, nmol=0, molecules=None):
        self.nmol = nmol
        if molecules is None:
            self.molecules = np.array([], dtype=object)
        else:
            self.molecules = np.array(molecules, dtype=object)

    def copy(self):
        """Returns a copy of the MoleculeBundle object."""
        new_mols = [mol.copy() for mol in self.molecules]
        return MoleculeBundle(np.copy(self.nmol), new_mols)

    def save(self):
        """Saves all molecules in the bundle."""
        for mol in self.molecules:
            mol.save()

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
        molecules = [new_molecules] if isinstance(new_molecules,
                                                  Molecule) else new_molecules
        self.nmol += len(molecules)
        self.molecules = np.hstack((self.molecules, molecules))

    def rm_molecules(self, ind):
        """Removes molecule(s) from the bundle by index."""
        self.nmol -= 1 if isinstance(ind, int) else len(ind)
        del self.molecules[ind]

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


def _rearrange_check(new_ind, old_ind, ntot):
    """Checks indices of rearrangement routines for errors."""
    if old_ind is None:
        if isinstance(new_ind, int) or len(new_ind) < ntot:
            raise ValueError('Old indices must be specified if length of '
                             'new indices less than natm')
        else:
            old_ind = range(ntot)
    if not isinstance(old_ind, type(new_ind)):
        raise TypeError('Old and new indices must be of the same type')
    elif isinstance(new_ind, list) and len(new_ind) != len(old_ind):
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

    return MoleculeBundle(len(molecules), molecules)
