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
4. X1 out-of-plane: r24 - r34 or (r24 x r34) x r14

Each internal coordinate measurement has the option of changing the units
(see the constants module) or taking the absolute value.
"""
import numpy as np
import geomtools.constants as con


def stre(xyz, ind, units='ang', absv=False):
    """Returns bond length based on index."""
    coord = np.linalg.norm(xyz[ind[0]] - xyz[ind[1]])
    return coord * con.conv('ang', units)


def bend(xyz, ind, units='rad', absv=False):
    """Returns bending angle for 3 atoms in a chain based on index."""
    e1 = xyz[ind[0]] - xyz[ind[1]]
    e2 = xyz[ind[2]] - xyz[ind[1]]
    e1 /= np.linalg.norm(e1)
    e2 /= np.linalg.norm(e2)

    coord = np.arccos(np.dot(e1, e2))
    return coord * con.conv('rad', units)


def tors(xyz, ind, units='rad', absv=False):
    """Returns dihedral angle for 4 atoms in a chain based on index."""
    e1 = xyz[ind[0]] - xyz[ind[1]]
    e2 = xyz[ind[2]] - xyz[ind[1]]
    e3 = xyz[ind[2]] - xyz[ind[3]]

    # get normals to 3-atom planes
    cp1 = np.cross(e1, e2)
    cp2 = np.cross(e2, e3)
    cp1 /= np.linalg.norm(cp1)
    cp2 /= np.linalg.norm(cp2)

    if absv:
        coord = np.arccos(np.dot(cp1, cp2))
    else:
        # get cross product of plane normals for signed dihedral angle
        cp3 = np.cross(cp1, cp2)
        coord = np.sign(np.dot(cp3, e2)) * np.arccos(np.dot(cp1, cp2))

    return coord * con.conv('rad', units)


def oop(xyz, ind, units='rad', absv=False):
    """Returns out-of-plane angle of atom 1 connected to atom 4 in the
    2-3-4 plane.

    Contains an additional sign convention such that rotation of the
    out-of-plane atom over (under) the central plane atom gives an angle
    greater than pi/2 (less than -pi/2).
    """
    e1 = xyz[ind[0]] - xyz[ind[3]]
    e2 = xyz[ind[1]] - xyz[ind[3]]
    e3 = xyz[ind[2]] - xyz[ind[3]]
    e1 /= np.linalg.norm(e1)
    e2 /= np.linalg.norm(e2)
    e3 /= np.linalg.norm(e3)

    sintau = np.dot(np.cross(e2, e3) / np.sqrt(1 - np.dot(e2, e3) ** 2), e1)
    coord = np.sign(np.dot(e2+e3, e1)) * np.arccos(sintau) + np.pi/2
    # sign convention to keep |oop| < pi
    if coord > np.pi:
        coord -= 2 * np.pi
    if absv:
        return abs(coord) * con.conv('rad', units)
    else:
        return coord * con.conv('rad', units)


def planeang(xyz, ind, units='rad', absv=False):
    """Returns the angle between two planes with 3 atoms each."""
    e1 = xyz[ind[0]] - xyz[ind[2]]
    e2 = xyz[ind[1]] - xyz[ind[2]]
    e3 = xyz[ind[3]] - xyz[ind[2]]
    e4 = xyz[ind[4]] - xyz[ind[3]]
    e5 = xyz[ind[5]] - xyz[ind[3]]

    # get normals to 3-atom planes
    cp1 = np.cross(e1, e2)
    cp2 = np.cross(e4, e5)
    cp1 /= np.linalg.norm(cp1)
    cp2 /= np.linalg.norm(cp2)

    if absv:
        coord = np.arccos(np.dot(cp1, cp2))
    else:
        # get cross product of plane norms for signed dihedral angle
        cp3 = np.cross(cp1, cp2)
        coord = np.sign(np.dot(cp3, e3)) * np.arccos(np.dot(cp1, cp2))

    return coord * con.conv('rad', units)


def planetors(xyz, ind, units='rad', absv=False):
    """Returns the plane angle with the central bond projected out."""
    e1 = xyz[ind[0]] - xyz[ind[2]]
    e2 = xyz[ind[1]] - xyz[ind[2]]
    e3 = xyz[ind[3]] - xyz[ind[2]]
    e4 = xyz[ind[4]] - xyz[ind[3]]
    e5 = xyz[ind[5]] - xyz[ind[3]]
    e3 /= np.linalg.norm(e3)

    # get normals to 3-atom planes
    cp1 = np.cross(e1, e2)
    cp2 = np.cross(e4, e5)

    # project out component along central bond
    pj1 = cp1 - np.dot(cp1, e3) * e3
    pj2 = cp2 - np.dot(cp2, e3) * e3
    pj1 /= np.linalg.norm(pj1)
    pj2 /= np.linalg.norm(pj2)

    if absv:
        coord = np.arccos(np.dot(pj1, pj2))
    else:
        # get cross product of vectors for signed dihedral angle
        cp3 = np.cross(pj1, pj2)
        coord = np.sign(np.dot(cp3, e3)) * np.arccos(np.dot(pj1, pj2))

    return coord * con.conv('rad', units)


def edgetors(xyz, ind, units='rad', absv=False):
    """Returns the torsional angle based on the vector difference of the
    two external atoms to the central bond"""
    e1 = xyz[ind[0]] - xyz[ind[2]]
    e2 = xyz[ind[1]] - xyz[ind[2]]
    e3 = xyz[ind[3]] - xyz[ind[2]]
    e4 = xyz[ind[4]] - xyz[ind[3]]
    e5 = xyz[ind[5]] - xyz[ind[3]]
    e1 /= np.linalg.norm(e1)
    e2 /= np.linalg.norm(e2)
    e3 /= np.linalg.norm(e3)
    e4 /= np.linalg.norm(e4)
    e5 /= np.linalg.norm(e5)

    # take the difference between unit vectors of external bonds
    e2 -= e1
    e5 -= e4

    # get cross products of difference vectors and the central bond
    cp1 = np.cross(e2, e3)
    cp2 = np.cross(e3, e5)
    cp1 /= np.linalg.norm(cp1)
    cp2 /= np.linalg.norm(cp2)

    if absv:
        coord = np.arccos(np.dot(cp1, cp2))
    else:
        # get cross product of vectors for signed dihedral angle
        cp3 = np.cross(cp1, cp2)
        coord = np.sign(np.dot(cp3, e3)) * np.arccos(np.dot(cp1, cp2))

    return coord * con.conv('rad', units)


def _parse_axis(inp):
    """Returns a numpy array based on a specified axis.

    Axis can be given as a string (e.g. 'x' or 'xy'), a vector
    or a set of 3 vectors. If the input defines a plane, the plane
    normal is returned.

    For instance, 'x', 'yz', [1, 0, 0] and [[0, 1, 0], [0, 0, 0], [0, 0, 1]]
    will all return [1, 0, 0].
    """
    if isinstance(inp, str):
        if inp in ['x', 'yz', 'zy']:
            return np.array([1., 0., 0.])
        elif inp in ['y', 'xz', 'zx']:
            return np.array([0., 1., 0.])
        elif inp in ['z', 'xy', 'yx']:
            return np.array([0., 0., 1.])
    elif len(inp) == 3:
        u = np.array(inp, dtype=float)
        if u.size == 9:
            unew = np.cross(u[0] - u[1], u[2] - u[1])
            return unew / np.linalg.norm(unew)
        else:
            return u / np.linalg.norm(u)
    else:
        raise ValueError('Axis specification not recognized')


def translate(xyz, amp, axis, ind=None, origin=np.zeros(3), units='ang'):
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


def refmat_ax(ax):
    """Returns the reflection matrix based on an axis.

    The reflection matrix in 3D about the plane with plane normal u is

    P = I - 2 u (x) u

    where I is the identity matrix and (x) is the outer product. Action
    of the reflection matrix occurs about the origin. See
    en.wikipedia.org/wiki/Transformation_matrix#Reflection_2
    """
    u = _parse_axis(ax)
    return np.eye(3) - 2*np.outer(u, u)


def ax_refmat(refmat):
    """Returns the axis of reflection based on the reflection matrix.

    The magnitude of u can be found from the diagonal elements of P by

    p_ii = 1 - 2 u_i^2
    |u_i| = sqrt((1 - p_ii) / 2)

    The sign can be found by letting sign(u_3) = +1 and solving

    sign(u_i) = sign(u_i) sign(u_3) = -sign(p_i3), i != 3

    Note that reflections along antiparallel axes are equivalent.
    """
    if not np.isclose(np.linalg.det(refmat), -1):
        raise ValueError('det(P) != -1, not a reflection matrix.')

    u = np.sqrt((1 - np.diag(refmat)) / 2)
    u[0] *= -np.sign(refmat[0,2])
    u[1] *= -np.sign(refmat[1,2])
    return u


def reflect(xyz, axis, ind=None, origin=np.zeros(3)):
    """Reflects a set of atoms across a given plane.

    An origin can be specified for reflection about a specific point. If
    no indices are specified, all atoms are displaced.
    """
    if ind is None:
        ind = range(len(xyz))
    origin = np.array(origin, dtype=float)
    newxyz = xyz - origin
    newxyz[ind] = np.dot(refmat_ax(axis), newxyz[ind].T).T
    return newxyz + origin


def rotmat_angax(ang, ax, units='rad'):
    """Returns the rotational matrix based on an angle and axis.

    The rotational matrix in 3D can be formed given an angle and an axis by

    R = cos(a) I + sin(a) [u]_x + (1 - cos(a)) u (x) u

    for identity matrix I, angle a, axis u, cross-product matrix [u]_x and
    outer product (x). Action of the rotational matrix occurs about the
    origin. See en.wikipedia.org/wiki/Rotation_matrix
    """
    u = _parse_axis(ax)
    amp = ang * con.conv(units, 'rad')
    ucross = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return (np.cos(amp) * np.eye(3) + np.sin(amp) * ucross +
            (1 - np.cos(amp)) * np.outer(u, u))


def angax_rotmat(rotmat, units='rad'):
    """Returns the angle and axis of rotation based on a rotational matrix.

    Based on the form of R, it can be separated into symmetric
    and antisymmetric components with (r_ij + r_ji)/2 and
    (r_ij - r_ji)/2, respectively. Then,

    r_ii = cos(a) + u_i^2 (1 - cos(a)),
    cos(a) = (-1 + sum_j r_jj) / 2 = (-1 + tr(R)) / 2.

    From the expression for r_ii, the magnitude of u_i can be found

    |u_i| = sqrt((1 + 2*r_ii - tr(R)) / 2),

    which satisfies u.u = 1. The sign can be found from the antisymmetric
    component of R

    u_i sin(a) = (r_jk - r_kj) / 2, i != j != k,
    sign(u_i) = sign(r_jk - r_kj)

    since sin(a) is positive in the range 0 to pi. i, j and k obey the
    cyclic relation 3 -> 2 -> 1 -> 3 -> ...

    This fails when a = pi, in which case the symmetric component of
    R is used

    u_i u_j (1 - cos(a)) = (r_ij + r_ji) / 2
    sign(u_i) sign(u_j) = sign(r_ij + r_ji)

    The signs can then be found by letting sign(u_3) = +1, since a rotation
    of pi is equivalent for antiparallel axes.

    Credit for angle derivation:
    euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/
    """
    if not np.isclose(np.linalg.det(rotmat), 1):
        raise ValueError('det(R) != 1, not a rotational matrix')

    tr = np.trace(rotmat)
    if np.isclose(tr, 3):
        return 0, np.array([0, 0, 1])

    u = np.sqrt((1 + 2*np.diag(rotmat) - tr) / (3 - tr))
    if np.isclose(tr, -1):
        u[1] *= np.sign(rotmat[1,2] + rotmat[2,1])
        u[0] *= np.sign(u[1]) * np.sign(rotmat[0,1] + rotmat[1,0])
        return np.pi * con.conv('rad', units), u
    else:
        u[0] *= np.sign(rotmat[2,1] - rotmat[1,2])
        u[1] *= np.sign(rotmat[0,2] - rotmat[2,0])
        u[2] *= np.sign(rotmat[1,0] - rotmat[0,1])
        return np.arccos((tr - 1) / 2) * con.conv('rad', units), u


def rotate(xyz, ang, axis, ind=None, origin=np.zeros(3), units='rad'):
    """Rotates a set of atoms about a given vector.

    An origin can be specified for rotation about a specific point. If
    no indices are specified, all atoms are displaced.
    """
    if ind is None:
        ind = range(len(xyz))
    origin = np.array(origin, dtype=float)
    newxyz = xyz - origin
    newxyz[ind] = np.dot(rotmat_angax(ang, axis, units=units), newxyz[ind].T).T
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
    test = _parse_axis(test_pl)
    ref = _parse_axis(ref_pl)
    return align_axis(xyz, test, ref, ind=ind, origin=origin)


def get_centremass(elem, xyz):
    """Returns centre of mass of a set of atoms."""
    mass = con.get_mass(elem)
    if isinstance(mass, float):
        # Centre of mass of one atom is its position
        return xyz
    else:
        return np.sum(mass[:,np.newaxis] * xyz, axis=0) / np.sum(mass)


def centre_mass(elem, xyz):
    """Returns xyz with centre of mass at the origin.

    If an index list is provided to inds, only the centre of mass of
    atoms at those indices will be used.
    """
    return xyz - get_centremass(elem, xyz)


def int_coord(xyz, funcs, axes, relamps=1, inds=None, origins=np.zeros(3)):
    """Defines a function that will return a displaced geometry based on
    a set of translations and rotations.

    At the moment, axes are in the space-fixed frame. Displacement about an
    atom that has been displaced could cause problems. The order in which
    funtions are specified is the order of operations.

    The relamps keyword should be changed to not depend on the units.
    """
    funcs = np.atleast_1d(funcs)
    nfuncs = len(funcs)
    funcnames = np.array([f.__name__ for f in funcs])
    if not np.all(np.logical_or(funcnames == 'translate',
                                funcnames == 'rotate')):
        raise ValueError('internal coordinates must be defined using '
                         '\'translate\' and \'rotate\' functions')
    functypes = np.zeros(nfuncs, dtype=int)
    functypes[funcnames == 'rotate'] = 1
    axes = np.atleast_2d(axes)

    if isinstance(relamps, (float,int)):
        relamps = np.ones(nfuncs)

    if inds is None:
        inds = np.tile(range(len(xyz)), (nfuncs, 1))
    elif isinstance(inds[0], int):
        inds = np.tile(inds, (nfuncs, 1))

    if isinstance(origins[0], (float,int)):
        origins = np.tile(origins, (nfuncs, 1))

    def _function(amp, units=['ang', 'rad']):
        newxyz = np.copy(xyz)
        for i, f in enumerate(funcs):
            newxyz = f(newxyz, amp*relamps[i], axes[i], inds[i],
                       origins[i], units[functypes[i]])
        return newxyz

    return _function


def int_path(coord, amin, amax, n=30, units=['ang', 'rad']):
    """Returns a list of xyz displaced along an internal coordinate."""
    alist = np.linspace(amin, amax, n)
    xyzlist = []
    for amp in alist:
        xyzlist.append(coord(amp, units=units))

    return np.array(xyzlist)


def int_grid(coords, amins, amaxs, n=30, units=['ang', 'rad']):
    """Returns a grid of xyz of arbitrary dimension displaced along
    internal coordinates.

    This requires some form of molecular frame axis specification.
    """
    pass
