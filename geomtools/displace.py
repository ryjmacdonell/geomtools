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
import numpy as np
import geomtools.constants as con


def stre(xyz, ind, units='ang'):
    """Returns bond length based on index."""
    coord = np.linalg.norm(xyz[ind[0]] - xyz[ind[1]])
    return coord * con.conv('ang', units)


def bend(xyz, ind, units='rad'):
    """Returns bending angle for 3 atoms in a chain based on index."""
    e1 = xyz[ind[0]] - xyz[ind[1]]
    e2 = xyz[ind[2]] - xyz[ind[1]]
    e1 /= np.linalg.norm(e1)
    e2 /= np.linalg.norm(e2)

    coord = np.arccos(np.dot(e1, e2))
    return coord * con.conv('rad', units)


def tors(xyz, ind, units='rad'):
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
    cp3 = np.cross(cp1, cp2)
    cp3 /= np.linalg.norm(cp3)

    coord = np.sign(np.dot(cp3, e2)) * np.arccos(np.dot(cp1, cp2))
    return coord * con.conv('rad', units)


def oop(xyz, ind, units='rad'):
    """Returns out-of-plane angle of atom 1 connected to atom 4 in the
    2-3-4 plane."""
    e1 = xyz[ind[0]] - xyz[ind[3]]
    e2 = xyz[ind[1]] - xyz[ind[3]]
    e3 = xyz[ind[2]] - xyz[ind[3]]
    e1 /= np.linalg.norm(e1)
    e2 /= np.linalg.norm(e2)
    e3 /= np.linalg.norm(e3)

    coord = np.arcsin(np.dot(np.cross(e2, e3) /
                             np.sqrt(1 - np.dot(e2, e3) ** 2), e1))
    return coord * con.conv('rad', units)


def translate(xyz, ind, amp, axis, origin=np.zeros(3), units='ang'):
    """Translates atoms given by ind along a vector u."""
    u = np.array(axis, dtype=float)
    u /= np.linalg.norm(u)
    origin = np.array(origin, dtype=float)
    amp *= con.conv(units, 'ang')

    newxyz = xyz - origin
    newxyz[ind] += amp * u
    return newxyz + origin


def rotate(xyz, ind, amp, axis, origin=np.zeros(3), units='rad'):
    """Rotates atoms given by ind about a vector u."""
    u = np.array(axis, dtype=float)
    u /= np.linalg.norm(u)
    origin = np.array(origin, dtype=float)
    amp *= con.conv(units, 'rad')
    uouter = np.outer(u, u)
    ucross = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    rotmat = (np.cos(amp) * np.eye(3) + np.sin(amp) * ucross +
              (1 - np.cos(amp)) * uouter)

    newxyz = xyz - origin
    newxyz[ind] = np.dot(rotmat, newxyz[ind].T).T
    return newxyz + origin


def combo(funcs, wgts=None):
    """Creates a combination function of translations and rotations.

    TODO: Find a better way to right this.
    """
    if wgts is None:
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


def get_centremass(elem, xyz):
    """Returns centre of mass of a set of atoms."""
    mass = con.get_mass(elem)
    return np.sum(mass[:,np.newaxis] * xyz, axis=0) / np.sum(mass)


def centre_mass(elem, xyz):
    """Returns xyz with centre of mass at the origin."""
    return xyz - get_centremass(elem, xyz)


def comment(s, func, inds):
    """Writes a comment line based on a measurement."""
    def _function(xyz):
        return s.format(func(xyz, inds))
    return _function


def c_loop(outfile, wfunc, disp, n, el, xyz, u, origin, ind, amplim,
           comm, namp):
    """Displaces by amplitudes in list and outputs geometries.

    TODO: Rewrite this.
    """
    amplist = np.linspace(amplim[0], amplim[1], namp)

    for amp in amplist:
        newxyz = disp(xyz, ind, amp, u, origin)
        wfunc(outfile, n, el, newxyz, comm(newxyz))


#if __name__ == '__main__':
#    import sys
#    fout = sys.stdout
#
#    fout.write('Tests for the python geometric displacement module.\n')
#
#    # basic test geometry
#    natm = 4
#    elem = ['B', 'C', 'N', 'O']
#    test_xyz = np.eye(4, 3)
#
#    # test translation
#    fout.write('\nTranslation by 1.0 Ang. along x axis:\n')
#    write_xyz(fout, natm, elem, translate(test_xyz, range(natm), 1.0, xyz[0]))
#
#    # test rotation
#    fout.write('\nRotation by pi/2 about x axis:\n')
#    write_xyz(fout, natm, elem, rotate(test_xyz, range(natm), np.pi/2, xyz[0]))
#
#    # test combination
#    fout.write('\nCombined translation by 1.0 Ang. and rotation by pi/2 '
#               'about x axis:\n')
#    write_xyz(fout, natm, elem, combo([translate, rotate], xyz, range(natm),
#                                      [1.0, np.pi/2], xyz[0]))
#
#    # test looping through geoms
#    fout.write('\nLooping atom C through pi/2 rotations about x axis:\n')
#    c_loop(fout, write_xyz, rotate, natm, elem, xyz, xyz[0], xyz[0], [1],
#           [np.pi/2, 2*np.pi], comment('CON angle: {:.4f} rad', bend,
#                                       [1, 3, 2]), 4)
