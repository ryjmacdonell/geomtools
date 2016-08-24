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


def _parse_axis(inp):
    """Returns a numpy array from string or axis vector."""
    if isinstance(inp, str):
        if inp == 'x':
            return np.array([1., 0., 0.])
        elif inp == 'y':
            return np.array([0., 1., 0.])
        elif inp == 'z':
            return np.array([0., 0., 1.])
    elif len(inp) == 3:
        u = np.array(inp, dtype=float)
        return u / np.linalg.norm(u)
    else:
        raise ValueError('Axis must be specified by cartesian axis or '
                         '3D vector.')


def _parse_plane(inp):
    """Returns a plane normal from a string, 3-atom set or plane normal."""
    if isinstance(inp, str):
        if inp == 'yz':
            return np.array([1., 0., 0.])
        elif inp == 'xz':
            return np.array([0., 1., 0.])
        elif inp == 'xy':
            return np.array([0., 0., 1.])
    elif len(inp) == 3:
        u = np.array(inp, dtype=float)
        if u.size == 9:
            return np.cross(u[0] - u[1], u[2] - u[1])
        else:
            return u
    else:
        raise ValueError('Plane must be specified by cartesian axes, plane '
                         'normal vector or set of 3 atoms.')


def translate(xyz, amp, axis, ind=None, units='ang'):
    """Translates a set of atoms along a given vector.

    If no indices are specified, all atoms are displaced.
    """
    if ind is None:
        ind = range(len(xyz))
    u = _parse_axis(axis)
    amp *= con.conv(units, 'ang')

    newxyz = np.copy(xyz)
    newxyz[ind] += amp * u
    return newxyz


def rotate(xyz, amp, axis, ind=None, origin=np.zeros(3), units='rad'):
    """Rotates a set of atoms about a given vector.

    The rotational matrix in 3D can be formed given an angle and an axis by

    R = cos(a) I + sin(a) [u]_x + (1 - cos(a)) u (x) u

    for angle a, axis u, cross-product matrix [u]_x and tensor product (x)
    (See en.wikipedia.org/wiki/Rotation_matrix). Action of the rotational
    matrix occurs about the origin, so an origin can be specified for
    rotation about a specific point.

    If no indices are specified, all atoms are displaced.
    """
    if ind is None:
        ind = range(len(xyz))
    u = _parse_axis(axis)
    origin = np.array(origin, dtype=float)
    amp *= con.conv(units, 'rad')
    ucross = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    uouter = np.outer(u, u)
    rotmat = (np.cos(amp) * np.eye(3) + np.sin(amp) * ucross +
              (1 - np.cos(amp)) * uouter)

    newxyz = xyz - origin
    newxyz[ind] = np.dot(rotmat, newxyz[ind].T).T
    return newxyz + origin


def align_pos(xyz, test_crd, ref_crd, ind=None):
    """Translates a set of atoms such that two positions are coincident."""
    transax = ref_crd - test_crd
    dist = np.linalg.norm(transax)
    return translate(xyz, dist, transax, ind=ind)


def align_axis(xyz, test_ax, ref_ax, ind=None, origin=np.zeros(3)):
    """Rotates a set of atoms such that two axes are parallel."""
    test = _parse_axis(test_ax)
    ref = _parse_axis(ref_ax)
    test /= np.linalg.norm(test)
    ref /= np.linalg.norm(ref)

    angle = np.arccos(np.dot(test, ref))
    rotax = np.cross(test, ref)
    return rotate(xyz, angle, rotax, ind=ind, origin=origin)


def align_plane(xyz, test_pl, ref_pl, ind=None, origin=np.zeros(3)):
    """Rotates a set of atoms such that two planes are parallel."""
    test = _parse_plane(test_pl)
    ref = _parse_plane(ref_pl)
    return align_axis(xyz, test, ref, ind=ind, origin=origin)


def get_centremass(elem, xyz):
    """Returns centre of mass of a set of atoms."""
    mass = con.get_mass(elem)
    if isinstance(mass, float):
        # Centre of mass of one atom is its position
        return xyz
    else:
        return np.sum(mass[:,np.newaxis] * xyz, axis=0) / np.sum(mass)


def centre_mass(elem, xyz, inds=None):
    """Returns xyz with centre of mass at the origin.

    If an index list is provided to inds, only the centre of mass of
    atoms at those indices will be used.
    """
    if inds is None:
        inds = range(len(elem))
    return xyz - get_centremass(elem[inds], xyz[inds])


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


def comment(s, func, inds):
    """Writes a comment line based on a measurement.

    TODO: Rewrite this.
    """
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
