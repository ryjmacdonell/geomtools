'''
Script for displacing a molecular geometry by translation (stretch) or rotation
(bend, torsion, out-of-plane motion).

Input
-----
Takes geometry in XYZ or COLUMBUS format. Indices and output types are given below. 
All displacements are made with respect to a given vector.

Output
------
May be XYZ (write_xyz) or COLUMBUS (write_col) format.

Notes
-----
        1
        |
        4
       / \
      2   3

Example axes for displacements:
1. X1X4 stretch: r14
2. X1X4 torsion: r14 (for motion of 2, 3)
3. X1X4X2 bend: r14 x r24
4. X1 out-of-plane: (r24 x r34) x r14
'''
import numpy as np

class Molecule(object):
    def __init__(self, natm=0, elem=np.array([], dtype=str), \
                xyz=np.array([], dtype=float).reshape(0, 3)):
        self.natm = natm
        self.elem = elem
        self.xyz = xyz
        self.save()

        self.a0 = 0.52917721092
        self.atmnum = {'X':0, 'H':1, 'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, \
            'O':8, 'F':9, 'Ne':10, 'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, \
            'S':16, 'Cl':17, 'Ar':18}
        self.atmmass = {'X':0.00000000, 'H':1.00782504, 'He':4.00260325, \
            'Li':7.01600450, 'Be':9.01218250, 'B':11.00930530, 'C':12.00000000, \
            'N':14.00307401, 'O':15.99491464, 'F':18.99840325, 'Ne':19.99243910, \
            'Na':22.98976970, 'Mg':23.98504500, 'Al':26.98154130, 'Si':27.97692840, \
            'P':30.97376340, 'S':31.97207180, 'Cl':34.96885273, 'Ar':39.96238310}

    def copy(self):
        # creates a copy of the molecule object
        return Molecule(np.copy(self.natm), np.copy(self.elem), np.copy(self.xyz))

    def save(self):
        # overwrites 'orig' properties with current properties
        self.orig_natm = np.copy(self.natm)
        self.orig_elem = np.copy(self.elem)
        self.orig_xyz = np.copy(self.xyz)
    
    def revert(self):
        # reverts properties to saved (orig) properties
        self.natm = np.copy(self.orig_natm)
        self.elem = np.copy(self.orig_elem)
        self.xyz = np.copy(self.orig_xyz)

    def add_atoms(self, new_elem, new_xyz):
        # adds atom(s) to molecule
        self.natm += 1 if isinstance(new_elem, str) else len(new_elem)
        self.elem = np.hstack((self.elem, new_elem))
        self.xyz = np.vstack((self.xyz, new_xyz))

    def rm_atoms(self, ind):
        # removes atom(s) from molecule
        self.natm -= 1 if isinstance(ind, int) else len(ind)
        self.elem = np.delete(self.elem, ind)
        self.xyz = np.delete(self.xyz, ind, axis=0)

    def read_xyz(self, fname):
        # reads input file in XYZ format
        fin = open(fname, 'r')

        self.natm = int(fin.readline())
        fin.readline()
        data = np.array([line.split() for line in fin.readlines()])
        fin.close()

        self.elem = data[:self.natm, 0]
        self.xyz = data[:self.natm, 1:].astype(float)
        self.save()

    def read_col(self, fname):
        # reads input file in COLUMBUS format
        fin = open(fname, 'r')

        data = np.array([line.split() for line in fin.readlines()])
        fin.close()

        self.natm = len(data)
        self.elem = data[:, 0]
        self.xyz = data[:, 2:-1].astype(float) * self.a0
        self.save()

    def write_col(self, outfile, comment=''):
        # outputs geometry in COLUMBUS format
        if comment != '':
            outfile.write('{}\n'.format(comment))

        for a, pos in zip(self.elem, self.xyz / self.a0):
            outfile.write(' {:<2}{:7.1f}{:14.8f}{:14.8f}{:14.8f}{:14.8f}\n'.format(\
                a, self.atmnum[a], pos[0], pos[1], pos[2], self.atmmass[a]))

    def write_xyz(self, outfile, comment=''):
        # outputs geometry in XYZ format
        outfile.write(' {}\n{}\n'.format(self.natm, comment))

        for a, pos in zip(self.elem, self.xyz):
            outfile.write('{:4s}{:12.6f}{:12.6f}{:12.6f}\n'.format(a, *pos))

    def translate(self, ind, c, u, orig=np.zeros(3)):
        # translates atoms given by 'ind' along a vector 'u'
        u /= np.linalg.norm(u)

        self.xyz -= orig
        self.xyz[ind] += c * u
        self.xyz + orig

    def rotate(self, ind, c, u, orig=np.zeros(3)):
        # rotates atoms given by 'ind' about a vector 'u'
        u /= np.linalg.norm(u)
        uouter = np.outer(u, u)
        ucross = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
        rotmat = np.cos(c) * np.eye(3) + np.sin(c) * ucross + (1 - np.cos(c)) * uouter

        self.xyz -= orig
        self.xyz[ind] = np.dot(rotmat, self.xyz[ind].T).T
        self.xyz += orig

    def get_stre(self, ind):
        # gets bond length between two atoms
        return np.linalg.norm(self.xyz[ind[0]] - self.xyz[ind[1]])

    def get_bend(self, ind):
        # gets bend angle for 3 atoms in a chain
        e1 = self.xyz[ind[0]] - self.xyz[ind[1]]
        e2 = self.xyz[ind[2]] - self.xyz[ind[1]]
        e1 /= np.linalg.norm(e1)
        e2 /= np.linalg.norm(e2)

        return np.arccos(np.dot(e1, e2))

    def get_tors(self, ind):
        # gets dihedral angle for a set of 4 atoms in a chain
        e1 = self.xyz[ind[0]] - self.xyz[ind[1]]
        e2 = self.xyz[ind[2]] - self.xyz[ind[1]]
        e3 = self.xyz[ind[2]] - self.xyz[ind[3]]
        e1 /= np.linalg.norm(e1)
        e2 /= np.linalg.norm(e2)
        e3 /= np.linalg.norm(e3)

        cp1 = np.cross(e1, e2)
        cp2 = np.cross(e2, e3)
        cp1 /= np.linalg.norm(cp1)
        cp2 /= np.linalg.norm(cp2)

        cp3 = np.cross(cp2, cp1)
        cp3 /= np.linalg.norm(cp3)

        return np.sign(np.dot(cp3, e2)) * np.arccos(np.dot(cp1, cp2))

    def get_oop(self, ind):
        # gets out-of-plane angle of atom 1 connected to atom 4 in the 2-3-4 plane
        e1 = self.xyz[ind[0]] - self.xyz[ind[3]]
        e2 = self.xyz[ind[1]] - self.xyz[ind[3]]
        e3 = self.xyz[ind[2]] - self.xyz[ind[3]]
        e1 /= np.linalg.norm(e1)
        e2 /= np.linalg.norm(e2)
        e3 /= np.linalg.norm(e3)

        return np.arcsin(np.dot(np.cross(e2, e3) / np.sqrt(1 - np.dot(e2, e3) ** 2), e1))

def read_xyz(fname):
    # reads input file in XYZ format
    fin = open(fname, 'r')

    natm = int(fin.readline())
    fin.readline()
    data = np.array([line.split() for line in fin.readlines()])
    fin.close()

    elem = data[:natm, 0]
    xyz = data[:natm, 1:].astype(float)

    return natm, elem, xyz

def read_col(fname):
    # reads input file in COLUMBUS format
    a0 = 0.52917721092
    fin = open(fname, 'r')

    data = np.array([line.split() for line in fin.readlines()])
    fin.close()

    natm = len(data)
    elem = data[:, 0]
    xyz = data[:, 2:-1].astype(float) * a0

    return natm, elem, xyz

def write_col(outfile, n, el, xyz, comment=''):
    # outputs geometry in COLUMBUS format
    a0 = 0.52917721092
    atmnum = {'X':0, 'H':1, 'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, \
        'F':9, 'Ne':10, 'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, \
        'Cl':17, 'Ar':18}
    atmmass = {'X':0.00000000, 'H':1.00782504, 'He':4.00260325, 'Li':7.01600450, \
        'Be':9.01218250, 'B':11.00930530, 'C':12.00000000, 'N':14.00307401, \
        'O':15.99491464, 'F':18.99840325, 'Ne':19.99243910, 'Na':22.98976970, \
        'Mg':23.98504500, 'Al':26.98154130, 'Si':27.97692840, 'P':30.97376340, \
        'S':31.97207180, 'Cl':34.96885273, 'Ar':39.96238310}

    outfile.write('{}\n'.format(comment))

    for a, pos in zip(el, xyz / a0): 
        outfile.write(' {:<2}{:7.1f}{:14.8f}{:14.8f}{:14.8f}{:14.8f}\n'.format(a, \
            atmnum[a], pos[0], pos[1], pos[2], atmmass[a]))

def write_xyz(outfile, n, el, xyz, comment=''):
    # outputs geometry in XYZ format
    outfile.write(' {}\n{}\n'.format(n, comment))

    for a, pos in zip(el, xyz):
        outfile.write('{:4s}{:12.6f}{:12.6f}{:12.6f}\n'.format(a, *pos))

def translate(xyz, ind, c, u, orig=np.zeros(3)):
    # translates atoms given by 'ind' along a vector 'u'
    u /= np.linalg.norm(u)

    newxyz = xyz - orig
    newxyz[ind] += c * u
    return newxyz + orig

def rotate(xyz, ind, c, u, orig=np.zeros(3)):
    # rotates atoms given by 'ind' about a vector 'u'
    u /= np.linalg.norm(u)
    uouter = np.outer(u, u)
    ucross = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    rotmat = np.cos(c) * np.eye(3) + np.sin(c) * ucross + (1 - np.cos(c)) * uouter

    newxyz = xyz - orig 
    newxyz[ind] = np.dot(rotmat, newxyz[ind].T).T
    return newxyz + orig

def combo(funcs, wgts=None):
    # creates a combination function of translations and rotations
    if wgts == None:
        wgts = np.ones(len(funcs))

    def _function(xyz, ind, c, u, orig=np.zeros(3)):
        newxyz = np.copy(xyz)
        to_list = [u, orig]
        [u, orig] = [s if isinstance(s, list) else [s] * len(funcs) for s in to_list]
        ind = ind if isinstance(ind[0], list) else [ind] * len(funcs)

        for i, f in enumerate(funcs):
           newxyz = f(newxyz, ind[i], c * wgts[i], u[i], orig[i]) 

        return newxyz

    return _function

def comment(s, func, inds):
    # writes comment line based on measurement
    def _function(xyz):
        return s.format(func(xyz, inds))

    return _function

def c_loop(outfile, wfunc, disp, n, el, xyz, u, origin, ind, clim, comm, nc=30):
    # displaces by amplitudes 'c' in list and outputs geometries
    clist = np.linspace(clim[0], clim[1], nc)

    for c in clist:
        newxyz = disp(xyz, ind, c, u, origin)
        wfunc(outfile, n, el, newxyz, comm(newxyz))

def stre(xyz, inds):
    # gets bond length between two atoms
    return np.linalg.norm(xyz[inds[0]] - xyz[inds[1]])

def bend(xyz, inds):
    # gets bend angle for 3 atoms in a chain
    e1 = xyz[inds[0]] - xyz[inds[1]]
    e2 = xyz[inds[2]] - xyz[inds[1]]
    e1 /= np.linalg.norm(e1)
    e2 /= np.linalg.norm(e2)

    return np.arccos(np.dot(e1, e2))

def tors(xyz, inds):
    # gets dihedral angle for a set of 4 atoms in a chain
    e1 = xyz[inds[0]] - xyz[inds[1]]
    e2 = xyz[inds[2]] - xyz[inds[1]]
    e3 = xyz[inds[2]] - xyz[inds[3]]
    e1 /= np.linalg.norm(e1)
    e2 /= np.linalg.norm(e2)
    e3 /= np.linalg.norm(e3)

    cp1 = np.cross(e1, e2)
    cp2 = np.cross(e2, e3)
    cp1 /= np.linalg.norm(cp1)
    cp2 /= np.linalg.norm(cp2)

    cp3 = np.cross(cp2, cp1)
    cp3 /= np.linalg.norm(cp3)
    
    return np.sign(np.dot(cp3, e2)) * np.arccos(np.dot(cp1, cp2))

def oop(xyz, inds):
    # gets out-of-plane angle of atom 1 connected to atom 4 in the 2-3-4 plane
    e1 = xyz[inds[0]] - xyz[inds[3]]
    e2 = xyz[inds[1]] - xyz[inds[3]]
    e3 = xyz[inds[2]] - xyz[inds[3]]
    e1 /= np.linalg.norm(e1)
    e2 /= np.linalg.norm(e2)
    e3 /= np.linalg.norm(e3)
    
    return np.arcsin(np.dot(np.cross(e2, e3) / np.sqrt(1 - np.dot(e2, e3) ** 2), e1))


if __name__ == '__main__':
    import sys
    fout = sys.stdout

    fout.write('Tests for the python geometric displacement module.\n')

    # basic test geometry
    natm = 4
    elem = ['B', 'C', 'N', 'O']
    xyz = np.eye(4, 3)

    # test output formats
    fout.write('\nStarting geometry (xyz format):\n')
    write_xyz(fout, natm, elem, xyz, comment='Comment line')
    fout.write('\nStarting geometry (COLUMBUS format):\n')
    write_col(fout, natm, elem, xyz, comment='Comment line')

    # test measurements
    fout.write('\n')
    fout.write('BO bond length: {:.4f} Angstroms\n'.format(stre(xyz, [0, 3])))
    fout.write('BOC bond angle: {:.4f} rad\n'.format(bend(xyz, [0, 3, 1])))
    fout.write('BOCN dihedral angle: {:.4f} rad\n'.format(tors(xyz, [0, 3, 1, 2])))
    fout.write('B-CON out-of-plane angle: {:.4f} rad\n'.format(oop(xyz, [0, 3, 1, 2])))

    # test translation
    fout.write('\nTranslation by 1.0 Ang. along x axis:\n')
    write_xyz(fout, natm, elem, translate(xyz, range(natm), 1.0, xyz[0]))

    # test rotation
    fout.write('\nRotation by pi/2 about x axis:\n')
    write_xyz(fout, natm, elem, rotate(xyz, range(natm), np.pi/2, xyz[0]))

    # test combination
    fout.write('\nCombined translation by 1.0 Ang. and rotation by pi/2 about x axis:\n')
    write_xyz(fout, natm, elem, combo([translate, rotate], xyz, range(natm), [1.0, np.pi/2], xyz[0]))

    # test looping through geoms
    fout.write('\nLooping atom C through pi/2 rotations about x axis:\n')
    c_loop(fout, write_xyz, rotate, natm, elem, xyz, xyz[0], xyz[0], [1], [np.pi/2, 2*np.pi], comment('CON angle: {:.4f} rad', bend, [1, 3, 2]), nc=4)
    
    print 'Exited successfully.'
