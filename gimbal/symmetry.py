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

    .. math::

        \mathbf{A} = (\mathbf{r} \cdot \mathbf{r}) \mathbf{I} -
        \mathbf{r} \otimes \mathbf{r}

    where **I** is the identity and :math:`\otimes` is an open product.
    The coordinates **r** are mass weighted cartesian coordinates,
    :math:`r_i = m_i^{1/2} q_i`

    Parameters
    ----------
    elem : (N,) array_like
        The atomic element list.
    xyz : (N, 3) array_like
        The cartesian coordinates of each atom.

    Returns
    -------
    (3,) ndarray
        The magnitudes of principal moments of inertia.
    (3, 3) ndarray
        The principal axes of inertia.
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

    Parameters
    ----------
    elem : (N,) array_like
        The atomic element list.
    xyz : (N, 3) array_like
        The cartesian coordinates of each atom.
    thresh : float, optional
        The tolerance threshold for symmetry recognition.

    Returns
    -------
    (N, 3) ndarray
        The cartesian coordinates of the symmetrized molecule.
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
    threshold.

    Parameters
    ----------
    xyz : (N, 3) array_like
        The set of atomic cartesian coordinates.
    thresh : float, optional
        The maximum random cartesian displacement.

    Returns
    -------
    (N, 3) ndarray
        The randomly displaced cartesian coordinates.
    """
    return xyz + thresh*np.random.rand(*xyz.shape)
