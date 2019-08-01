"""
Routines for displacing a molecular geometry by translation or
proper/improper rotation.

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
import gimbal.constants as con


def translate(xyz, amp, axis, ind=None, units='ang'):
    """Translates a set of atoms along a given vector.

    Parameters
    ----------
    xyz : (N, 3) array_like
        The atomic cartesian coordinates.
    amp : float
        The distance for translation.
    axis : array_like or str
        The axis of translation, parsed by :func:`_parse_axis`.
    ind : array_like, optional
        List of atomic indices to specify which atoms are displaced. If
        ind is None (default) then all atoms are displaced.
    units : str, optional
        The units of length for displacement. Default is angstroms.

    Returns
    -------
    (N, 3) ndarray
        The atomic cartesian coordinates of the displaced molecule.
    """
    if ind is None:
        ind = range(len(xyz))
    u = _parse_axis(axis)
    amp *= con.conv(units, 'ang')

    newxyz = np.copy(xyz)
    newxyz[ind] += amp * u
    return newxyz


def rotmat(ang, ax, det=1, units='rad'):
    """Returns the rotational matrix based on an angle and axis.

    A general rotational matrix in 3D can be formed given an angle and
    an axis by

    R = cos(a) I + (det(R) - cos(a)) u (x) u + sin(a) [u]_x

    for identity matrix I, angle a, axis u, outer product (x) and
    cross-product matrix [u]_x. Determinants of +1 and -1 give proper
    and improper rotation, respectively. Thus, det(R) = -1 and a = 0
    is a reflection along the axis. Action of the rotational matrix occurs
    about the origin. See en.wikipedia.org/wiki/Rotation_matrix
    and http://scipp.ucsc.edu/~haber/ph251/rotreflect_17.pdf

    Parameters
    ----------
    ang : float
        The angle of rotation.
    ax : array_like or str
        The axis of rotation, parsed by :func:`_parse_axis`.
    det : int, optional
        The determinant of the matrix (1 or -1) used to specify proper
        and improper rotations. Default is 1.
    units : str, optional
        The units of angle for the rotation. Default is radians.

    Returns
    -------
    (3, 3) ndarray
        The rotational matrix of the given angle and axis.

    Raises
    ------
    ValueError
        When the absolute value of the determinant is not equal to 1.
    """
    if not np.isclose(np.abs(det), 1):
        raise ValueError('Determinant of a rotational matrix must be +/- 1')

    u = _parse_axis(ax)
    amp = ang * con.conv(units, 'rad')
    ucross = np.array([[0, u[2], -u[1]], [-u[2], 0, u[0]], [u[1], -u[0], 0]])
    return (np.cos(amp) * np.eye(3) + np.sin(amp) * ucross +
            (det - np.cos(amp)) * np.outer(u, u))


def angax(rotmat, units='rad'):
    """Returns the angle, axis of rotation and determinant of a
    rotational matrix.

    Based on the form of R, it can be separated into symmetric
    and antisymmetric components with (r_ij + r_ji)/2 and
    (r_ij - r_ji)/2, respectively. Then,

    r_ii = cos(a) + u_i^2 (det(R) - cos(a)),
    cos(a) = (-det(R) + sum_j r_jj) / 2 = (tr(R) - det(R)) / 2.

    From the expression for r_ii, the magnitude of u_i can be found

    |u_i| = sqrt((1 + det(R) [2 r_ii - tr(R)]) / 2),

    which satisfies u.u = 1. Note that if det(R) tr(R) = 3, the axis
    is arbitrary (identity or inversion). Otherwise, the sign can be found
    from the antisymmetric component of R

    u_i sin(a) = (r_jk - r_kj) / 2, i != j != k,
    sign(u_i) = sign(r_jk - r_kj),

    since sin(a) is positive in the range 0 to pi. i, j and k obey the
    cyclic relation 3 -> 2 -> 1 -> 3 -> ...

    This fails when det(R) tr(R) = -1, in which case the symmetric
    component of R is used

    u_i u_j (det(R) - cos(a)) = (r_ij + r_ji) / 2,
    sign(u_i) sign(u_j) = det(R) sign(r_ij + r_ji).

    The signs can then be found by letting sign(u_3) = +1, since a rotation
    of pi or a reflection are equivalent for antiparallel axes. See
    http://scipp.ucsc.edu/~haber/ph251/rotreflect_17.pdf

    Parameters
    ----------
    rotmat : (3, 3) array_like
        The rotational matrix.
    units : str, optional
        The output units for the angle. Default is radians.

    Returns
    -------
    ang : float
        The angle of rotation.
    u : (3,) ndarray
        The axis of rotation as a 3D vector.
    det : int
        The determinant of the rotation matrix.

    Raises
    ------
    ValueError
        When the absolute value of the determinant is not equal to 1.
    """
    det = np.linalg.det(rotmat)
    if not np.isclose(np.abs(det), 1):
        raise ValueError('Determinant of a rotational matrix must be +/- 1')

    tr = np.trace(rotmat)
    ang = np.arccos((tr - det) / 2) * con.conv('rad', units)
    if np.isclose(det*tr, 3):
        u = np.array([0, 0, 1])
    else:
        u = np.sqrt((1 + det*(2*np.diag(rotmat) - tr)) / (3 - det*tr))
        if np.isclose(det*tr, -1):
            sgn = np.ones(3)
            sgn[1] = det * _nonzero_sign(rotmat[1,2] + rotmat[2,1])
            sgn[0] = det * sgn[1] * _nonzero_sign(rotmat[0,1] + rotmat[1,0])
            u *= sgn
        else:
            u[0] *= _nonzero_sign(rotmat[1,2] - rotmat[2,1])
            u[1] *= _nonzero_sign(rotmat[2,0] - rotmat[0,2])
            u[2] *= _nonzero_sign(rotmat[0,1] - rotmat[1,0])

    return ang, u, det


def rotate(xyz, ang, axis, ind=None, origin=np.zeros(3), det=1, units='rad'):
    """Rotates a set of atoms about a given vector.

    An origin can be specified for rotation about a specific point. If
    no indices are specified, all atoms are displaced. Setting det=-1
    leads to an improper rotation.

    Parameters
    ----------
    xyz : (N, 3) array_like
        The atomic cartesian coordinates.
    ang : float
        The angle of rotation.
    axis : array_like or str
        The axis of rotation, parsed by :func:`_parse_axis`.
    ind : array_like, optional
        List of atomic indices to specify which atoms are displaced. If
        ind is None (default) then all atoms are displaced.
    origin : (3,) array_like, optional
        The origin of rotation. Default is the cartesian origin.
    det : float, optional
        The determinant of the rotation. 1 (default) is a proper rotation
        and -1 is an improper rotation (rotation + reflection).
    units : str, optional
        The units of length for displacement. Default is angstroms.

    Returns
    -------
    (N, 3) ndarray
        The atomic cartesian coordinates of the displaced molecule.
    """
    if ind is None:
        ind = range(len(xyz))
    origin = np.array(origin, dtype=float)
    newxyz = xyz - origin
    newxyz[ind] = newxyz[ind].dot(rotmat(ang, axis, det=det, units=units))
    return newxyz + origin


def align_pos(xyz, test_crd, ref_crd, ind=None):
    """Translates a set of atoms such that two positions are coincident.

    Parameters
    ----------
    xyz : (N, 3) array_like
        The atomic cartesian coordinates.
    test_crd : (3,) array_like
        Cartesian coordinates of the original position.
    test_crd : (3,) array_like
        Cartesian coordinates of the final position.
    ind : array_like, optional
        List of atomic indices to specify which atoms are displaced. If
        `ind == None` (default) then all atoms are displaced.

    Returns
    -------
    (N, 3) ndarray
        The atomic cartesian coordinates of the displaced molecule.
    """
    transax = ref_crd - test_crd
    dist = np.linalg.norm(transax)
    return translate(xyz, dist, transax, ind=ind)


def align_axis(xyz, test_ax, ref_ax, ind=None, origin=np.zeros(3)):
    """Rotates a set of atoms such that two axes are parallel.

    Parameters
    ----------
    xyz : (N, 3) array_like
        The atomic cartesian coordinates.
    test_crd : (3,) array_like
        Cartesian coordinates of the original axis.
    test_crd : (3,) array_like
        Cartesian coordinates of the final axis.
    ind : array_like, optional
        List of atomic indices to specify which atoms are displaced. If
        `ind == None` (default) then all atoms are displaced.
    origin : (3,) array_like, optional
        The origin of rotation. Default is the cartesian origin.

    Returns
    -------
    (N, 3) ndarray
        The atomic cartesian coordinates of the displaced molecule.
    """
    test = _parse_axis(test_ax)
    ref = _parse_axis(ref_ax)
    if np.allclose(test, ref):
        return xyz
    elif np.allclose(test, -ref):
        rotax = np.array([0., 0., 1.])
        if np.allclose(test, rotax) or np.allclose(test, -rotax):
            rotax = np.array([0., 1., 0.])
        rotax -= np.dot(rotax, test) * test
        return rotate(xyz, np.pi, rotax, ind=ind, origin=origin)
    else:
        angle = np.arccos(np.dot(test, ref))
        rotax = np.cross(test, ref)
        return rotate(xyz, angle, rotax, ind=ind, origin=origin)


def get_centremass(elem, xyz):
    """Returns centre of mass of a set of atoms.

    Parameters
    ----------
    elem : (N,) array_like
        The atomic symbols.
    xyz : (N, 3) array_like
        The atomic cartesian coordinates.

    Returns
    -------
    (3,) ndarray
        The position of the centre of mass.
    """
    mass = con.get_mass(elem)
    if isinstance(mass, float):
        # Centre of mass of one atom is its position
        return xyz
    elif np.allclose(mass, mass[0]):
        # If masses are identical (including zero), return mean position
        return np.sum(xyz, axis=0) / len(xyz)
    else:
        return np.sum(mass[:,np.newaxis] * xyz, axis=0) / np.sum(mass)


def centre_mass(elem, xyz):
    """Returns xyz with centre of mass at the origin.

    If an index list is provided to inds, only the centre of mass of
    atoms at those indices will be used.

    Parameters
    ----------
    elem : (N,) array_like
        The atomic symbols.
    xyz : (N, 3) array_like
        The atomic cartesian coordinates.

    Returns
    -------
    (N, 3) ndarray
        The atomic cartesian coordinates of the displaced molecule.
    """
    return xyz - get_centremass(elem, xyz)


def _parse_axis(inp):
    """Returns a numpy array based on a specified axis.

    Axis can be given as a string (e.g. 'x' or 'xy'), a vector
    or a set of 3 vectors. If the input defines a plane, the plane
    normal is returned.

    For instance, 'x', 'yz', [1, 0, 0] and [[0, 1, 0], [0, 0, 0], [0, 0, 1]]
    will all return [1, 0, 0].

    Parameters
    ----------
    inp : array_like or str
        The axis specification to be parsed.

    Returns
    -------
    (3,) ndarray
        The axis given as a 3D cartesian vector.
    """
    if isinstance(inp, str):
        if inp in ['x', 'yz', 'zy']:
            return np.array([1., 0., 0.])
        elif inp in ['y', 'xz', 'zx']:
            return np.array([0., 1., 0.])
        elif inp in ['z', 'xy', 'yx']:
            return np.array([0., 0., 1.])
        elif inp == '-x':
            return np.array([-1., 0., 0.])
        elif inp == '-y':
            return np.array([0., -1., 0.])
        elif inp == '-z':
            return np.array([0., 0., -1.])
    elif len(inp) == 3:
        u = np.array(inp, dtype=float)
        if u.size == 9:
            unew = np.cross(u[0] - u[1], u[2] - u[1])
            return con.unit_vec(unew)
        else:
            return con.unit_vec(u)
    else:
        raise ValueError('Axis specification not recognized')


def _nonzero_sign(x):
    """Returns the sign of a nonzero number, otherwise returns 1.

    Parameters
    ----------
    x : float or int
        The input number.

    Returns
    -------
    float
        The sign of x, i.e. +1 or -1.
    """
    if np.isclose(x, 0.):
        return 1.
    else:
        return np.sign(x)
