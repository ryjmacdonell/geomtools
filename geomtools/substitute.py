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
        self.elem = dict()
        self.xyz = dict()
        self.syn = dict()
        self._populate_elem()
        self._populate_xyz()
        self._populate_syn()

    def _populate_elem(self):
        """Adds element labels to self.elem."""
        self.elem['ch3'] = np.array(['C', 'H', 'H', 'H'])
        self.elem['chch2'] = np.array(['C', 'C', 'H', 'H', 'H'])
        self.elem['cch'] = np.array(['C', 'C', 'H'])
        self.elem['nh2'] = np.array(['N', 'H', 'H'])
        self.elem['chnh'] = np.array(['C', 'N', 'H', 'H'])
        self.elem['cn'] = np.array(['C', 'N'])
        self.elem['oh'] = np.array(['O', 'H'])
        self.elem['cho'] = np.array(['C', 'O', 'H'])
        self.elem['no2'] = np.array(['N', 'O', 'O'])
        self.elem['f'] = np.array(['F'])
        self.elem['sh'] = np.array(['S', 'H'])
        self.elem['sh'] = np.array(['S', 'H'])
        self.elem['so2h'] = np.array(['S', 'O', 'O', 'H'])
        self.elem['cl'] = np.array(['Cl'])

    def _populate_xyz(self):
        """Adds cartesian geometries to self.xyz.

        For all substituents, the bonding atom is at the origin,
        the bonding axis is the z-axis and the plane axis
        is the y-axis.
        """
        self.xyz['ch3'] = np.array([[0.000, 0.000, 0.000],
                                    [-1.023, 0.000, 0.377],
                                    [0.511, -0.886, 0.377],
                                    [0.511, 0.886, 0.377]])
        self.xyz['chch2'] = np.array([[0.000, 0.000, 0.000],
                                      [-1.124, 0.000,  0.730],
                                      [0.971, 0.000, 0.495],
                                      [-2.095, 0.000, 0.235],
                                      [-1.067, 0.000, 1.818]])
        self.xyz['cch'] = np.array([[0.000, 0.000, 0.000],
                                    [0.000, 0.000, 1.210],
                                    [0.000, 0.000, 2.280]])
        self.xyz['nh2'] = np.array([[0.000, 0.000, 0.000],
                                    [-0.577, -0.771,  0.332],
                                    [-0.577, 0.771,  0.332]])
        self.xyz['chnh'] = np.array([[0.000, 0.000, 0.000],
                                     [-1.082, 0.000, 0.703],
                                     [0.980, 0.000, 0.499],
                                     [-0.869, 0.000, 1.710]])
        self.xyz['cn'] = np.array([[0.000, 0.000, 0.000],
                                   [0.000, 0.000, 1.136]])
        self.xyz['oh'] = np.array([[0.000, 0.000, 0.000],
                                   [-0.913, 0.000, 0.297]])
        self.xyz['cho'] = np.array([[0.000, 0.000, 0.000],
                                    [-1.011, 0.000, 0.700],
                                    [0.998, 0.000, 0.463]])
        self.xyz['no2'] = np.array([[0.000, 0.000, 0.000],
                                    [-1.105, 0.000, 0.563],
                                    [1.105, 0.000, 0.563]])
        self.xyz['f'] = np.array([[0.000, 0.000, 0.000]])
        self.xyz['sh'] = np.array([[0.000, 0.000, 0.000],
                                   [-1.331, 0.000, 0.156]])
        self.xyz['so2h'] = np.array([[0.000, 0.000, 0.000],
                                     [0.548, -1.266, 0.448],
                                     [0.548, 1.266, 0.448],
                                     [-1.311, 0.000, 0.279]])
        self.xyz['cl'] = np.array([[0.000, 0.000, 0.000]])

    def _populate_syn(self):
        """Adds a dictionary of synonyms for labels."""
        synlist = [['ch3', 'h3c', 'me'],
                   ['chch2', 'c2h3', 'h2chc', 'h3c2', 'vi'],
                   ['cch', 'c2h', 'hcc', 'hc2', 'ey'],
                   ['nh2', 'h2n', 'am'],
                   ['chnh', 'cnh2', 'nhch', 'im'],
                   ['cn', 'nc'],
                   ['oh', 'ho'],
                   ['cho', 'coh', 'och', 'ohc', 'al'],
                   ['no2', 'o2n', 'nt'],
                   ['f'],
                   ['sh', 'hs'],
                   ['so2h', 'sooh', 'sho2', 'ho2s', 'hso2'],
                   ['cl']]
        for subl in synlist:
            for item in subl:
                self.syn[item] = subl[0]

    def get_sub(self, label):
        lbl = self.syn[label.lower()]
        return self.elem[lbl], self.xyz[lbl]


def import_sub(label):
    """Returns the element list and cartesian geometry of a substituent
    given its label."""
    lib = SubLib()
    return lib.get_sub(label)


def subst(elem, xyz, sublbl, isub, ibond, pl=None, mom=None):
    """Returns a molecular geometry with an specified atom replaced by
    substituent.

    Labels are case-insensitive. Indices isub and ibond give
    the position to be substituted, the position that will be bonded to
    the substituent (i.e. the axis). The orientation of the substituent
    can be given as a vector (the plane normal) or an index (the plane
    containing isub, ibond and pl).

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
    if pl is None:
        # choose an arbitrary axis and project out the bond axis
        pl = np.array([1., 1., 1.])
        pl -= np.dot(pl, ax) * ax
    elif isinstance(pl, int):
        pl = np.cross(xyz[ipos] - xyz[ibond], xyz[pl] - xyz[ibond])

    sub_el, sub_xyz = import_sub(sublbl)
    if elem[ipos] == sub_el[0]:
        blen = np.linalg.norm(xyz[ipos] - xyz[ibond])
    else:
        blen = con.get_covrad(elem[ibond]) + con.get_covrad(sub_el[0])

    # rotate to correct orientation
    sub_xyz = displace.align_axis(sub_xyz, 'z', ax)
    sub_pl = displace.align_axis([0., 1., 0.], 'z', ax)
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
