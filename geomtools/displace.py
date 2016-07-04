"""
Script for displacing a molecular geometry by translation (stretch) or rotation
(bend, torsion, out-of-plane motion).

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
"""
import sys
import numpy as np


def translate(xyz, ind, amp, u, orig=np.zeros(3)):
    """Translates atoms given by ind along a vector u."""
    u /= np.linalg.norm(u)

    newxyz = xyz - orig
    newxyz[ind] += amp * u
    return newxyz + orig


def rotate(xyz, ind, amp, u, orig=np.zeros(3)):
    """Rotates atoms given by ind about a vector u."""
    u /= np.linalg.norm(u)
    uouter = np.outer(u, u)
    ucross = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    rotmat = np.cos(amp) * np.eye(3) + np.sin(amp) * ucross + 
             (1 - np.cos(amp)) * uouter

    newxyz = xyz - orig 
    newxyz[ind] = np.dot(rotmat, newxyz[ind].T).T
    return newxyz + orig


def combo(funcs, wgts=None):
    """Creates a combination function of translations and rotations."""
    if wgts == None:
        wgts = np.ones(len(funcs))

    def _function(xyz, ind, amp, u, orig=np.zeros(3)):
        newxyz = np.copy(xyz)
        to_list = [u, orig]
        [u, orig] = [s if isinstance(s, list) else [s] * len(funcs) 
                     for s in to_list]
        ind = ind if isinstance(ind[0], list) else [ind] * len(funcs)

        for i, f in enumerate(funcs):
           newxyz = f(newxyz, ind[i], amp * wgts[i], u[i], orig[i]) 
        return newxyz
    return _function


def comment(s, func, inds):
    """Writes a comment line based on a measurement."""
    def _function(xyz):
        return s.format(func(xyz, inds))
    return _function


def c_loop(outfile, wfunc, disp, n, el, xyz, u, origin, ind, amplim, comm, n):
    """Displaces by amplitudes in list and outputs geometries."""
    amplist = np.linspace(amplim[0], amplim[1], n)

    for amp in amplist:
        newxyz = disp(xyz, ind, amp, u, origin)
        wfunc(outfile, n, el, newxyz, comm(newxyz))


if __name__ == '__main__':
    fout = sys.stdout

    fout.write('Tests for the python geometric displacement module.\n')

    # basic test geometry
    natm = 4
    elem = ['B', 'C', 'N', 'O']
    xyz = np.eye(4, 3)

    # test translation
    fout.write('\nTranslation by 1.0 Ang. along x axis:\n')
    write_xyz(fout, natm, elem, translate(xyz, range(natm), 1.0, xyz[0]))

    # test rotation
    fout.write('\nRotation by pi/2 about x axis:\n')
    write_xyz(fout, natm, elem, rotate(xyz, range(natm), np.pi/2, xyz[0]))

    # test combination
    fout.write('\nCombined translation by 1.0 Ang. and rotation by pi/2 '
               'about x axis:\n')
    write_xyz(fout, natm, elem, combo([translate, rotate], xyz, range(natm), 
              [1.0, np.pi/2], xyz[0]))

    # test looping through geoms
    fout.write('\nLooping atom C through pi/2 rotations about x axis:\n')
    c_loop(fout, write_xyz, rotate, natm, elem, xyz, xyz[0], xyz[0], [1], 
           [np.pi/2, 2*np.pi], comment('CON angle: {:.4f} rad', bend, 
                                       [1, 3, 2]), 4)
