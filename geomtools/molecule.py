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
        """Removes atom(s) from molecule based on index."""
        self.natm -= 1 if isinstance(ind, int) else len(ind)
        self.elem = np.delete(self.elem, ind)
        self.xyz = np.delete(self.xyz, ind, axis=0)
        self._check()
        self.saved = False

    def rearrange(self, new_ind, old_ind=None):
        """Moves atom(s) from old_ind to new_ind."""
        if old_ind is None:
            if isinstance(new_ind, int) or len(new_ind) < self.natm:
                raise ValueError('Old indices must be specified if length of '
                                 'new indices less than natm')
            else:
                old_ind = range(self.natm)
        if not isinstance(old_ind, type(new_ind)):
            raise TypeError('Old and new indices must be of the same type')
        elif isinstance(new_ind, list) and len(new_ind) != len(old_ind):
            raise IndexError('Old and new indices must be the same length')

        self.xyz[old_ind] = self.xyz[new_ind]

    # Input
    def read_xyz(self, fname, hc=False):
        """Reads input file in XYZ format."""
        with open(fname, 'r') as infile:
            (self.natm, self.elem, self.xyz,
             self.comment) = fileio.read_xyz(infile, hascomment=hc)
        self.save()

    def read_col(self, fname, hc=False):
        """Reads input file in COLUMBUS format."""
        with open(fname, 'r') as infile:
            (self.natm, self.elem, self.xyz,
             self.comment) = fileio.read_col(infile, hascomment=hc)
        self.save()

    def read_zmt(self, fname, hc=False):
        """Reads input file in Z-matrix format."""
        with open(fname, 'r') as infile:
            (self.natm, self.elem, self.xyz,
             self.comment) = fileio.read_zmt(infile, hascomment=hc)
        self.save()

    # Output
    def write_xyz(self, outfile):
        """Writes geometry to an output file in XYZ format."""
        fileio.write_xyz(outfile, self.natm, self.elem, self.xyz, self.comment)

    def write_col(self, outfile):
        """Writes geometry to an output file in COLUMBUS format."""
        fileio.write_col(outfile, self.natm, self.elem, self.xyz, self.comment)

    def write_zmt(self, outfile):
        """Writes geometry to an output file in Z-matrix format."""
        fileio.write_zmt(outfile, self.natm, self.elem, self.xyz, self.comment)

    def write_zmtvar(self, outfile):
        """Writes geometry to an output file in Z-matrix format
        with assignment variables."""
        fileio.write_zmtvar(outfile, self.natm, self.elem, self.xyz,
                            self.comment)

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


def Bundle(object):
    """
    Object containing a set of molecules in the form of Molecule
    objects.
    """
    def __init__(self, nmol=0, molecules=[]):
        self.nmol = nmol
        self.molecules = molecules

    def copy(self):
        """Returns a copy of the Bundle object."""
        return [mol.copy() for mol in self.molecules]

    def save(self):
        """Saves all molecules in the molecule bundle."""
        for mol in molecules:
            mol.save()

    def revert(self):
        """Reverts each molecule in the molecule bundle."""
        for mol in molecules:
            mol.revert()


def import_xyz(fname, hc=False):
    """Imports geometry in XYZ format to Molecule object."""
    with open(fname, 'r') as infile:
        return Molecule(*fileio.read_xyz(infile, hascomment=hc))


def import_col(fname, hc=False):
    """Imports geometry in COLUMBUS format to Molecule object."""
    with open(fname, 'r') as infile:
        return Molecule(*fileio.read_col(infile, hascomment=hc))


def import_zmt(fname, hc=False):
    """Imports geometry in Z-matrix format to Molecule object."""
    with open(fname, 'r') as infile:
        return Molecule(*fileio.read_zmt(infile, hascomment=hc))


if __name__ == '__main__':
    import sys
    fout = sys.stdout
    fout.write('Tests for the Python molecular geometry module.\n')

    # basic test geometry
    natm = 4
    elem = ['B', 'C', 'N', 'O']
    xyz = np.eye(4, 3)
    test = Molecule(natm, elem, xyz, 'Comment line')

    # test output formats
    fout.write('\nGeometry in XYZ format:\n')
    test.write_xyz(fout)
    fout.write('\nGeometry in COLUMBUS format:\n')
    test.write_col(fout)
    fout.write('\nGeometry in Z-matrix format:\n')
    test.write_zmt(fout)

    # test measurements
    fout.write('\n')
    fout.write('BO bond length: {:.4f} Angstroms'
               '\n'.format(test.get_stre([0, 3])))
    fout.write('BOC bond angle: {:.4f} rad'
               '\n'.format(test.get_bend([0, 3, 1])))
    fout.write('BOCN dihedral angle: {:.4f} rad'
               '\n'.format(test.get_tors([0, 3, 1, 2])))
    fout.write('B-CON out-of-plane angle: {:.4f} rad'
               '\n'.format(test.get_oop([0, 3, 1, 2])))

    # test adding/removing atoms
    fout.write('\nAdding F atom:\n')
    test.add_atoms('F', [-1, 0, 0])
    test.write_xyz(fout)
    fout.write('\nRemoving B atom:\n')
    test.rm_atoms(0)
    test.write_xyz(fout)
    fout.write('\nSwitching atoms 1 and 3\n')
    test.rearrange(0, 3)
    test.write_xyz(fout)
    fout.write('\nReverting to original geometry\n')
    test.revert()
    test.write_xyz(fout)

    fout.close()
