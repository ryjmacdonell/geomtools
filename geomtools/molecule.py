"""
The Molecule object and tools for generating and querying molecular geometries.

Input and output in XYZ, COLUMBUS or ZMAT formats. Creates a saved copy of the 
geometry after input for reversion after an operation.

TODO: Finish ZMAT I/O. Add custom formats and support for internal coordinates.
"""
import sys
import numpy as np


class Molecule(object):
    """
    Object containing the molecular geometry and functions for setting and
    getting geometric properties.
    """
    def __init__(self, natm=0, elem=np.array([], dtype=str),
                 xyz=np.array([], dtype=float).reshape(0, 3)):
        self.natm = natm
        self.elem = elem
        self.xyz = xyz
        self.save()

        # constants
        self.a0 = 0.52917721092
        self.atmnum = {'X':0, 'H':1, 'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6,
                       'N':7, 'O':8, 'F':9, 'Ne':10, 'Na':11, 'Mg':12, 'Al':13,
                       'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18}
        self.atmmass = {'X':0.00000000, 'H':1.00782504, 'He':4.00260325,
                        'Li':7.01600450, 'Be':9.01218250, 'B':11.00930530,
                        'C':12.00000000, 'N':14.00307401, 'O':15.99491464,
                        'F':18.99840325, 'Ne':19.99243910, 'Na':22.98976970,
                        'Mg':23.98504500, 'Al':26.98154130, 'Si':27.97692840,
                        'P':30.97376340, 'S':31.97207180, 'Cl':34.96885273, 
                        'Ar':39.96238310}


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
        self.natm = np.copy(self.orig_natm)
        self.elem = np.copy(self.orig_elem)
        self.xyz = np.copy(self.orig_xyz)
        self.saved = True


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


    def read_xyz(self, fname):
        """Reads input file in XYZ format."""
        with open(fname, 'r') as fin:
            self.natm = int(fin.readline())
            fin.readline()
            data = np.array([line.split() for line in fin.readlines()])

        self.elem = data[:self.natm, 0]
        self.xyz = data[:self.natm, 1:].astype(float)
        self.save()


    def read_col(self, fname):
        """Reads input file in COLUMBUS format."""
        with open(fname, 'r') as fin:
            data = np.array([line.split() for line in fin.readlines()])

        self.natm = len(data)
        self.elem = data[:, 0]
        self.xyz = data[:, 2:-1].astype(float) * self.a0
        self.save()


    def read_zmat(self, fname):
        """Reads input file in ZMAT format."""
        pass # this might require importing the displacement module


    def write_xyz(self, outfile, comment=''):
        """Writes geometry to an output file in XYZ format."""
        outfile.write(' {}\n{}\n'.format(self.natm, comment))

        for a, pos in zip(self.elem, self.xyz):
            outfile.write('{:4s}{:12.6f}{:12.6f}{:12.6f}\n'.format(a, *pos))


    def write_col(self, outfile, comment=''):
        """Writes geometry to an output file in COLUMBUS format."""
        if comment != '':
            outfile.write('{}\n'.format(comment))

        for a, pos in zip(self.elem, self.xyz / self.a0):
            outfile.write(' {:<2}{:7.1f}{:14.8f}{:14.8f}{:14.8f}{:14.8f}'
                          '\n'.format(a, self.atmnum[a], *pos, self.atmmass[a]))


    def write_zmat(self, outfile, comment=''):
        """Writes geometry to an output file in ZMAT format."""
        pass # this is relatively easy


    def get_stre(self, ind):
        return stre(self.xyz, ind)

    
    def get_bend(self, ind):
        return bend(self.xyz, ind)


    def get_tors(self, ind):
        return tors(self.xyz, ind)


    def get_oop(self, ind):
        return oop(self.xyz, ind)


def stre(xyz, ind):
    """Returns bond length based on index."""
    return np.linalg.norm(xyz[ind[0]] - xyz[ind[1]])


def bend(xyz, ind):
    """Returns bending angle for 3 atoms in a chain based on index."""
    e1 = xyz[ind[0]] - xyz[ind[1]]
    e2 = xyz[ind[2]] - xyz[ind[1]]
    e1 /= np.linalg.norm(e1)
    e2 /= np.linalg.norm(e2)

    return np.arccos(np.dot(e1, e2))


def tors(xyz, ind):
    """Returns dihedral angle for 4 atoms in a chain based on index."""
    e1 = xyz[ind[0]] - xyz[ind[1]]
    e2 = xyz[ind[2]] - xyz[ind[1]]
    e3 = xyz[ind[2]] - xyz[ind[3]]
    e1 /= np.linalg.norm(e1)
    e2 /= np.linalg.norm(e2)
    e3 /= np.linalg.norm(e3)

    # get normals to 3-atom planes
    cp1 = np.cross(e1, e2)
    cp2 = np.cross(e2, e3)
    cp1 /= np.linalg.norm(cp1)
    cp2 /= np.linalg.norm(cp2)

    # get cross product of plane normals for signed dihedral angle
    cp3 = np.cross(cp2, cp1)
    cp3 /= np.linalg.norm(cp3)

    return np.sign(np.dot(cp3, e2)) * np.arccos(np.dot(cp1, cp2))


def oop(xyz, ind):
    """Returns out-of-plane angle of atom 1 connected to atom 4 in the
    2-3-4 plane."""
    e1 = xyz[ind[0]] - xyz[ind[3]]
    e2 = xyz[ind[1]] - xyz[ind[3]]
    e3 = xyz[ind[2]] - xyz[ind[3]]
    e1 /= np.linalg.norm(e1)
    e2 /= np.linalg.norm(e2)
    e3 /= np.linalg.norm(e3)

    return np.arcsin(np.dot(np.cross(e2, e3) / 
                            np.sqrt(1 - np.dot(e2, e3) ** 2), e1))


if __name__ == '__main__':
    fout = sys.stdout
    fout.write('Tests for the Python molecular geometry module.\n')

    # basic test geometry
    natm = 4
    elem = ['B', 'C', 'N', 'O']
    xyz = np.eye(4, 3)
    test = Molecule(natm, elem, xyz)

    # test output formats
    fout.write('\nGeometry in XYZ format:\n')
    test.write_xyz(fout, comment='Comment line')
    fout.write('\nGeometry in COLUMBUS format:\n')
    test.write_col(fout, comment='Comment line')

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
    fout.write('\nReverting to original geometry\n')
    test.revert()
    test.write_xyz(fout)

    fout.close()
