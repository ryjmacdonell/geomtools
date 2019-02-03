"""
Routines for measuring internal coordinate values from a cartesian
geometry.
"""
import numpy as np
import geomtools.constants as con


def stre(xyz, *inds, units='ang', absv=False):
    """Returns bond length based on index."""
    coord = np.linalg.norm(xyz[inds[0]] - xyz[inds[1]])
    return coord * con.conv('ang', units)


def bend(xyz, *inds, units='rad', absv=False):
    """Returns bending angle for 3 atoms in a chain based on index."""
    e1 = con.unit_vec(xyz[inds[0]] - xyz[inds[1]])
    e2 = con.unit_vec(xyz[inds[2]] - xyz[inds[1]])

    coord = np.arccos(np.dot(e1, e2))
    return coord * con.conv('rad', units)


def tors(xyz, *inds, units='rad', absv=False):
    """Returns dihedral angle for 4 atoms in a chain based on index."""
    e1 = xyz[inds[0]] - xyz[inds[1]]
    e2 = xyz[inds[2]] - xyz[inds[1]]
    e3 = xyz[inds[2]] - xyz[inds[3]]

    # get normals to 3-atom planes
    cp1 = con.unit_vec(np.cross(e1, e2))
    cp2 = con.unit_vec(np.cross(e2, e3))

    if absv:
        coord = np.arccos(np.dot(cp1, cp2))
    else:
        # get cross product of plane normals for signed dihedral angle
        cp3 = np.cross(cp1, cp2)
        coord = np.sign(np.dot(cp3, e2)) * np.arccos(np.dot(cp1, cp2))

    return coord * con.conv('rad', units)


def oop(xyz, *inds, units='rad', absv=False):
    """Returns out-of-plane angle of atom 1 connected to atom 4 in the
    2-3-4 plane.

    Contains an additional sign convention such that rotation of the
    out-of-plane atom over (under) the central plane atom gives an angle
    greater than pi/2 (less than -pi/2).
    """
    e1 = con.unit_vec(xyz[inds[0]] - xyz[inds[3]])
    e2 = con.unit_vec(xyz[inds[1]] - xyz[inds[3]])
    e3 = con.unit_vec(xyz[inds[2]] - xyz[inds[3]])

    sintau = np.dot(np.cross(e2, e3) / np.sqrt(1 - np.dot(e2, e3) ** 2), e1)
    coord = np.sign(np.dot(e2+e3, e1)) * np.arccos(sintau) + np.pi/2
    # sign convention to keep |oop| < pi
    if coord > np.pi:
        coord -= 2 * np.pi
    if absv:
        return abs(coord) * con.conv('rad', units)
    else:
        return coord * con.conv('rad', units)


def planeang(xyz, *inds, units='rad', absv=False):
    """Returns the angle between the 1-2-3 and 4-5-6 planes."""
    e1 = xyz[inds[0]] - xyz[inds[2]]
    e2 = xyz[inds[1]] - xyz[inds[2]]
    e3 = xyz[inds[3]] - xyz[inds[2]]
    e4 = xyz[inds[4]] - xyz[inds[3]]
    e5 = xyz[inds[5]] - xyz[inds[3]]

    # get normals to 3-atom planes
    cp1 = con.unit_vec(np.cross(e1, e2))
    cp2 = con.unit_vec(np.cross(e4, e5))

    if absv:
        coord = np.arccos(np.dot(cp1, cp2))
    else:
        # get cross product of plane norms for signed dihedral angle
        cp3 = np.cross(cp1, cp2)
        coord = np.sign(np.dot(cp3, e3)) * np.arccos(np.dot(cp1, cp2))

    return coord * con.conv('rad', units)


def planetors(xyz, *inds, units='rad', absv=False):
    """Returns the plane angle with the central bond projected out."""
    e1 = xyz[inds[0]] - xyz[inds[2]]
    e2 = xyz[inds[1]] - xyz[inds[2]]
    e3 = con.unit_vec(xyz[inds[3]] - xyz[inds[2]])
    e4 = xyz[inds[4]] - xyz[inds[3]]
    e5 = xyz[inds[5]] - xyz[inds[3]]

    # get normals to 3-atom planes
    cp1 = np.cross(e1, e2)
    cp2 = np.cross(e4, e5)

    # project out component along central bond
    pj1 = con.unit_vec(cp1 - np.dot(cp1, e3) * e3)
    pj2 = con.unit_vec(cp2 - np.dot(cp2, e3) * e3)

    if absv:
        coord = np.arccos(np.dot(pj1, pj2))
    else:
        # get cross product of vectors for signed dihedral angle
        cp3 = np.cross(pj1, pj2)
        coord = np.sign(np.dot(cp3, e3)) * np.arccos(np.dot(pj1, pj2))

    return coord * con.conv('rad', units)


def edgetors(xyz, *inds, units='rad', absv=False):
    """Returns the torsional angle based on the vector difference of the
    two external atoms (1-2 and 5-6) to the central 3-4 bond."""
    e1 = con.unit_vec(xyz[inds[0]] - xyz[inds[2]])
    e2 = con.unit_vec(xyz[inds[1]] - xyz[inds[2]])
    e3 = con.unit_vec(xyz[inds[3]] - xyz[inds[2]])
    e4 = con.unit_vec(xyz[inds[4]] - xyz[inds[3]])
    e5 = con.unit_vec(xyz[inds[5]] - xyz[inds[3]])

    # take the difference between unit vectors of external bonds
    e2 -= e1
    e5 -= e4

    # get cross products of difference vectors and the central bond
    cp1 = con.unit_vec(np.cross(e2, e3))
    cp2 = con.unit_vec(np.cross(e3, e5))

    if absv:
        coord = np.arccos(np.dot(cp1, cp2))
    else:
        # get cross product of vectors for signed dihedral angle
        cp3 = np.cross(cp1, cp2)
        coord = np.sign(np.dot(cp3, e3)) * np.arccos(np.dot(cp1, cp2))

    return coord * con.conv('rad', units)
