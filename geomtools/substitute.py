"""
Substitution of molecular geometries with functional groups.

Given a cartesian geometry and element labels, a substituent can
be added knowing (a) the substituent identity (b) the desired
position to substitute and (c) the bond axis for substitution.

This requires some default information, such as the default structure
and orientation of the substituent (relative to an axis) and the
bond length of the substituent. For now, only single bonds are treated.
"""
import numpy as np
import geomtools.displace as displace
import geomtools.fileio as fileio
import geomtools.constants as con


class SubLib(object):
    """
    Object containing a library of substituent geometries.
    """
    def __init__(self):
        self.subs = ['cn', 'ch3']
        self.elem = dict()
        self.xyz = dict()
        self._populate_elem()
        self._populate_xyz()

    def _populate_elem(self):
        """Adds element labels to self.elem."""
        self.elem['cn'] = np.array(['C', 'N'])
        self.elem['ch3'] = np.array(['C', 'H', 'H', 'H'])

    def _populate_xyz(self):
        """Adds cartesian geometries to self.xyz."""
        self.xyz['cn'] = np.array([[0.000, 0.000, 0.000],
                                   [0.000, 0.000, 1.136]])
        self.xyz['ch3'] = np.array([[0.000, 0.000, 0.000],
                                    [-1.023, 0.000, 0.377],
                                    [0.511, -0.886, 0.377],
                                    [0.511, 0.886, 0.377]])

    def _parse_label(self, label):
        """Returns a label in a single format."""
        return label.lower()

    def get_sub(self, label):
        lbl = self._parse_label(label)
        return self.elem[lbl], self.xyz[lbl]


def import_sub(label):
    """Returns the element list and cartesian geometry of a substituent
    given its label."""
    lib = SubLib()
    return lib.get_sub(label)


def subst(elem, xyz, sublbl, isub, ibond, iplane=None, mom=None):
    """Returns a molecular geometry with an specified atom replaced by
    substituent.

    Labels are case-insensitive. Indices isub, ibond and iplane give
    the position to be substituted, the position that will be bonded to
    the substituent (i.e. the axis) and an optional 3rd index to define
    the plane (if any) of the substituent.

    If isub is given as a list, the entire list of atoms is be removed
    and the first index is treated as the position of the substituent.
    """
    elem = np.array(elem)
    xyz = np.atleast_2d(xyz)
    if not isinstance(isub, int):
        ipos = isub[0]
    else:
        isub = [isub]
        ipos = isub[0]

    ax = xyz[ipos] - xyz[ibond]
    ax /= np.linalg.norm(ax)
    origin = xyz[ibond]
    if iplane is None:
        pl = np.array([0, 1, 0])
        pl -= np.dot(pl, ax)
    else:
        pl = np.cross(xyz[ipos] - xyz[ibond], xyz[iplane] - xyz[ibond])

    sub_el, sub_xyz = import_sub(sublbl)
    blen = con.get_covrad(elem[ibond]) + con.get_covrad(sub_el[0])

    # rotate to correct orientation
    sub_xyz = displace.align_axis(sub_xyz, 'z', ax)
    sub_pl = displace.align_axis([0, 1, 0], 'z', ax)
    sub_xyz = displace.align_axis(sub_xyz, sub_pl, pl)

    # displace to correct position
    sub_xyz += xyz[ibond]
    sub_xyz += blen * ax

    # build the final geometry
    ind1 = [i for i in range(ipos) if i not in isub[1:]]
    ind2 = [i for i in range(ipos+1, len(elem)) if i not in isub[1:]]
    new_elem = np.hstack((elem[ind1], sub_el, elem[ind2]))
    new_xyz = np.vstack((xyz[ind1], sub_xyz, xyz[ind2]))
    if mom is None:
        new_mom = np.zeros((len(new_elem), 3))
    else:
        new_mom = np.vstack((mom[ind1], np.zeros((len(sub_el), 3)), mom[ind2]))
    return new_elem, new_xyz, new_mom
