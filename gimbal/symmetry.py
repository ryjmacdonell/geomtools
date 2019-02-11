"""
Routines to detect, symmetrize and generate symmetric geometries.

The symmetric axes are automatically found by the principal axes of
rotation for convenience. As a result, some axes need to be rearranged
to match normal conventions depending on the type of rotor.
"""
import numpy as np
import gimbal.constants as con
import gimbal.displace as displace


def principal_axes(elem, xyz):
    """Returns the moments of inertia and corresponding principal axes of
    a geometry.

    The moments of inertia are found by diagonalizing the inertia tensor,

    A = r . r I - r (x) r

    where I is the identity, . is a dot product and (x) is an open product.
    The coordinates r are mass weighted cartesian coordinates,
    r_i = sqrt(m_i) q_i
    """
    rxyz = np.sqrt(con.get_mass(elem))[:,np.newaxis] * xyz
    inert = np.sum(rxyz**2) * np.eye(3) - rxyz.T.dot(rxyz)
    return np.linalg.eig(inert)


def assign_rotor(moms):
    """Returns the type of rotor based on the moments of inertia."""
    pass


def symmetrize(elem, xyz, thresh=1e-3):
    """Returns a geometry and its point group with symmetry defined by a
    geometric threshold.

    The output geometry is rotated to have symmetry elements along
    cartesian axes if possible.
    """
    # centre and align
    new_xyz = displace.centre_mass(elem, xyz)
    moms, eigv = principal_axes(elem, new_xyz)
    eigv[:,-1] *= np.linalg.det(eigv)
    new_xyz = new_xyz.dot(eigv)

    # find atoms of the same type that are equidistant from the origin
    dist = np.linalg.norm(new_xyz, axis=1)
    #...
    return new_xyz


def cart_jumble(xyz, thresh=1e-3):
    """Moves cartesian coordinates by random amounts up to a given
    threshold."""
    return xyz + thresh*np.random.rand(*xyz.shape)
