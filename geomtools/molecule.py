"""
The Molecule and MoleculeBundle objects and tools for generating and
querying molecular geometries.

Molecule creates a saved copy of the geometry after input for reversion
after an operation. Can add/remove individual atoms or groups or set the
full geometry.

Likewise, MoleculeBundle creates a saved copy of a set molecular
geometries. Input files with multiple geometries can be read to a bundle.
"""
import numpy as np
import geomtools.fileio as fileio
import geomtools.displace as displace
import geomtools.constants as con


class BaseMolecule(object):
    """
    Basic object containing molecular geometry and functions for setting and
    changing the geometry.

    All methods of BaseMolecule involve setting and saving the molecular
    geometry. There are no dependancies to other geomtools modules.
    """
    def __init__(self, elem=np.array([], dtype=str), xyz=np.empty((0, 3)),
                 comment=''):
        self.elem = np.array(elem, dtype=str)
        self.xyz = np.array(xyz, dtype=float)
        self.comment = comment
        self.natm = len(elem)
        self.saved = True
        self.save()

    def _check(self):
        """Checks that xyz is 3D and len(elem) = len(xyz)."""
        if self.xyz.shape[1] != 3:
            raise ValueError('Molecular geometry must be 3-dimensional.')
        len_elem = len(self.elem)
        len_xyz = len(self.xyz)
        if len_elem != len_xyz:
            raise ValueError('Number of element labels ({:d}) not equal '
                             'to number of cartesian vectors '
                             '({:d}).'.format(len_elem, len_xyz))

    def copy(self, comment=None):
        """Creates a copy of the BaseMolecule object."""
        self._check()
        if comment is None:
            comment = 'Copy of ' + self.comment
        return BaseMolecule(np.copy(self.elem[1:]), np.copy(self.xyz[1:]),
                            comment)

    def save(self):
        """Saves molecular properties to 'orig' variables."""
        self._check()
        self.orig_elem = np.copy(self.elem)
        self.orig_xyz = np.copy(self.xyz)
        self.orig_comment = np.copy(self.comment)
        self.saved = True

    def revert(self):
        """Reverts properties to 'orig' variables."""
        if not self.saved:
            self.elem = np.copy(self.orig_elem)
            self.xyz = np.copy(self.orig_xyz)
        self.saved = True

    def set_geom(self, elem, xyz):
        """Sets molecular geometry."""
        if elem is not None:
            self.elem = elem
        self.xyz = np.array(xyz, dtype=float)
        self._check()
        self.saved = False

    def set_comment(self, comment):
        """Adds a comment line to describe the molecule."""
        self.comment = comment
        self.saved = False

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
        if old_ind is None:
            old_ind = range(self.natm)
        _rearrange_check(new_ind, old_ind)
        old = np.hstack((new_ind, old_ind))
        new = np.hstack((old_ind, new_ind))
        self.xyz[old] = self.xyz[new]
        self.elem[old] = self.elem[new]
        self.saved = False


class Molecule(BaseMolecule):
    """
    More advanced Molecule object which inherits from BaseMolecule.

    A dummy atom XM is added to index 0 at the centre of mass of the
    molecule. Indices thus match the regular definition. Accessor methods
    get_elem and get_xyz return the geometry without the index 0 dummy atom.

    Molecule can also read from and write to input files given by a filename
    or an open file object. Other methods are derived from geomtools modules.
    """
    def _add_centre(self):
        """Adds a dummy atom at index 0 at molecular centre of mass."""
        pos = displace.get_centremass(self.elem, self.xyz)
        if self.elem[0] == 'XM':
            self.xyz[0] = pos
        else:
            self.elem = np.hstack(('XM', self.elem))
            self.xyz = np.vstack((pos, self.xyz))

    def _check(self):
        """Checks that xyz is 3D and len(elem) = len(xyz) and that dummy
        atom is at centre of mass."""
        BaseMolecule._check(self)
        if self.natm > 0:
            self._add_centre()

    def copy(self, comment=None):
        """Creates a copy of the Molecule object."""
        self._check()
        if comment is None:
            comment = 'Copy of ' + self.comment
        return Molecule(np.copy(self.elem[1:]), np.copy(self.xyz[1:]),
                        comment)

    # Input/Output
    def read(self, infile, fmt='auto', hc=False):
        """Reads single geometry from input file in provided format."""
        read_func = getattr(fileio, 'read_' + fmt)
        if isinstance(infile, str):
            with open(infile, 'r') as f:
                self.elem, self.xyz, self.comment = read_func(f, hascomment=hc)
        else:
            self.elem, self.xyz, self.comment = read_func(infile, hascomment=hc)
        self.natm = len(self.elem)
        self.save()

    def write(self, outfile, fmt='auto'):
        """Writes geometry to an output file in provided format."""
        write_func = getattr(fileio, 'write_' + fmt)
        if isinstance(outfile, str):
            with open(outfile, 'w') as f:
                write_func(f, self.elem[1:], self.xyz[1:], self.comment)
        else:
            write_func(outfile, self.elem[1:], self.xyz[1:], self.comment)

    # Accessors
    def get_natm(self):
        """Returns number of atoms."""
        return self.natm

    def get_elem(self):
        """Returns list of elements."""
        return self.elem[1:]

    def get_xyz(self):
        """Returns cartesian geometry."""
        return self.xyz[1:]

    def get_comment(self):
        """Returns comment line."""
        return self.comment

    def get_mass(self):
        """Returns atomic masses."""
        return con.get_mass(self.elem[1:])

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

    def get_planeang(self, ind, units='rad'):
        """Returns plane angle based on index in molecule."""
        return displace.planeang(self.xyz, ind, units=units)

    def get_planetors(self, ind, units='rad'):
        """Returns plane dihedral angle based on index in molecule."""
        return displace.planetors(self.xyz, ind, units=units)

    # Displacement
    def centre_mass(self):
        """Places the centre of mass at the origin."""
        self.xyz -= self.xyz[0]
        self.saved = False

    def translate(self, amp, axis, ind=None, units='ang'):
        """Translates the molecule along a given axis."""
        self.xyz = displace.translate(self.xyz, amp, axis, ind=ind,
                                      units=units)
        self._add_centre()
        self.saved = False

    def rotate(self, amp, axis, ind=None, origin=np.zeros(3), units='rad'):
        """Rotates the molecule about a given axis from a given origin."""
        self.xyz = displace.rotate(self.xyz, amp, axis, ind=ind,
                                   origin=origin, units=units)
        self._add_centre()
        self.saved = False


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
        if old_ind is None:
            old_ind = range(self.nmol)
        _rearrange_check(new_ind, old_ind)
        old = np.hstack((new_ind, old_ind))
        new = np.hstack((old_ind, new_ind))
        self.molecules[old] = self.molecules[new]

    def add_molecules(self, new_molecules):
        """Adds molecule(s) to the bundle."""
        self.nmol += (1 if isinstance(new_molecules, Molecule)
                      else len(new_molecules))
        self.molecules = np.hstack((self.molecules, new_molecules))
        self._check()

    def rm_molecules(self, ind):
        """Removes molecule(s) from the bundle by index."""
        self.nmol -= 1 if isinstance(ind, int) else len(ind)
        self.molecules = np.delete(self.molecules, ind)

    # Input/Output
    def read(self, infile, fmt='auto', hc=False):
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

    def write(self, outfile, fmt='auto'):
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


def _rearrange_check(new_ind, old_ind):
    """Checks indices of rearrangement routines for errors."""
    new = [new_ind] if isinstance(new_ind, int) else new_ind
    old = [old_ind] if isinstance(old_ind, int) else old_ind
    if len(new) != len(old):
        raise IndexError('Old and new indices must be the same length')


def import_molecule(fname, fmt='auto', hc=False):
    """Imports geometry in provided format to Molecule object."""
    read_func = getattr(fileio, 'read_' + fmt)
    with open(fname, 'r') as infile:
        return Molecule(*read_func(infile, hascomment=hc))


def import_bundle(fname, fmt='auto', hc=False):
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
