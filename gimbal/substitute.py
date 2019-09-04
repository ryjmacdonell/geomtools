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
import gimbal.displace as displace
import gimbal.fileio as fileio
import gimbal.constants as con


class SubLib(object):
    """
    Object containing a library of substituent geometries.

    Attributes
    ----------
    syn : dict
        A dictionary of synonyms used to find the appropriate
        substituents.
    elem : dict
        A dictionary of atomic symbols for each substituent.
    xyz : dict
        A dictionary of atomic cartesian coordinates for each
        substituent.
    """
    def __init__(self):
        self.syn = dict()
        self.elem = dict()
        self.xyz = dict()
        self._populate_syn()
        self._populate_elem()
        self._populate_xyz()
        self._add_comb()

    def _populate_syn(self):
        """Adds a dictionary of synonyms for labels."""
        synlist = [['h'],
                   ['d'],
                   ['me', 'ch3', 'h3c'],
                   ['et', 'ch2ch3', 'c2h5', 'ch3ch2'],
                   ['npr', 'ch2ch2ch3', 'c3h7', 'ch3ch2ch2'],
                   ['ipr', 'chch3ch3', 'ch(ch3)2', '(ch3)2ch', 'ch3chch3'],
                   ['nbu', 'ch2ch2ch2ch3', 'c4h9', 'ch3ch2ch2ch2'],
                   ['ibu', 'chch3ch2ch3', 'ch3chch2ch3', 'ch3ch3chch3'],
                   ['tbu', 'cch3ch3ch3', 'c(ch3)3', 'ch3ch3ch3c', '(ch3)3c'],
                   ['vi', 'chch2', 'c2h3', 'h2chc', 'h3c2'],
                   ['ey', 'cch', 'c2h', 'hcc', 'hc2'],
                   ['ph', 'c6h5', 'h5c6'],
                   ['am', 'nh2', 'h2n'],
                   ['im', 'chnh', 'cnh2', 'nhch'],
                   ['cn', 'nc'],
                   ['oh', 'ho'],
                   ['ome', 'meo', 'och3', 'ch3o'],
                   ['al', 'cho', 'coh', 'och', 'ohc'],
                   ['ac', 'coch3', 'cch3o'],
                   ['ca', 'cooh', 'co2h', 'hooc', 'ho2c'],
                   ['nt', 'no2', 'o2n'],
                   ['f'],
                   ['tfm', 'cf3', 'f3c'],
                   ['sh', 'hs'],
                   ['sf', 'so2h', 'sooh', 'sho2', 'ho2s', 'hso2'],
                   ['ms', 'sfme', 'mesf', 'sfch3', 'so2me', 'so2ch3'],
                   ['cl']]
        for subl in synlist:
            for item in subl:
                self.syn[item] = subl[0]

    def _populate_elem(self):
        """Adds element labels to self.elem."""
        self.elem['h'] = np.array(['H'])
        self.elem['d'] = np.array(['D'])
        self.elem['me'] = np.array(['C', 'H', 'H', 'H'])
        self.elem['vi'] = np.array(['C', 'C', 'H', 'H', 'H'])
        self.elem['ey'] = np.array(['C', 'C', 'H'])
        self.elem['ph'] = np.array(['C', 'C', 'C', 'C', 'C', 'C',
                                    'H', 'H', 'H', 'H', 'H'])
        self.elem['am'] = np.array(['N', 'H', 'H'])
        self.elem['im'] = np.array(['C', 'N', 'H', 'H'])
        self.elem['cn'] = np.array(['C', 'N'])
        self.elem['oh'] = np.array(['O', 'H'])
        self.elem['al'] = np.array(['C', 'O', 'H'])
        self.elem['nt'] = np.array(['N', 'O', 'O'])
        self.elem['f'] = np.array(['F'])
        self.elem['sh'] = np.array(['S', 'H'])
        self.elem['sf'] = np.array(['S', 'O', 'O', 'H'])
        self.elem['cl'] = np.array(['Cl'])

    def _populate_xyz(self):
        """Adds cartesian geometries to self.xyz.

        For all substituents, the bonding atom is at the origin,
        the bonding axis is the z-axis and the plane axis
        is the y-axis.
        """
        self.xyz['h'] = np.array([[ 0.000,  0.000,  0.000]])
        self.xyz['d'] = np.array([[ 0.000,  0.000,  0.000]])
        self.xyz['me'] = np.array([[ 0.000,  0.000,  0.000],
                                   [ 0.511, -0.886,  0.377],
                                   [ 0.511,  0.886,  0.377],
                                   [-1.023,  0.000,  0.377]])
        self.xyz['vi'] = np.array([[ 0.000,  0.000,  0.000],
                                   [-1.124,  0.000,  0.730],
                                   [ 0.971,  0.000,  0.495],
                                   [-1.067,  0.000,  1.818],
                                   [-2.095,  0.000,  0.235]])
        self.xyz['ey'] = np.array([[ 0.000,  0.000,  0.000],
                                   [ 0.000,  0.000,  1.210],
                                   [ 0.000,  0.000,  2.280]])
        self.xyz['ph'] = np.array([[ 0.000,  0.000,  0.000],
                                   [-1.212,  0.000,  0.700],
                                   [ 1.212,  0.000,  0.700],
                                   [-1.212,  0.000,  2.100],
                                   [ 1.212,  0.000,  2.100],
                                   [ 0.000,  0.000,  2.800],
                                   [-2.156,  0.000,  0.155],
                                   [ 2.156,  0.000,  0.155],
                                   [-2.156,  0.000,  2.645],
                                   [ 2.156,  0.000,  2.645],
                                   [ 0.000,  0.000,  3.890]])
        self.xyz['am'] = np.array([[ 0.000,  0.000,  0.000],
                                   [-0.577, -0.771,  0.332],
                                   [-0.577,  0.771,  0.332]])
        self.xyz['im'] = np.array([[ 0.000,  0.000,  0.000],
                                   [-1.082,  0.000,  0.703],
                                   [ 0.980,  0.000,  0.499],
                                   [-0.869,  0.000,  1.710]])
        self.xyz['cn'] = np.array([[ 0.000,  0.000,  0.000],
                                   [ 0.000,  0.000,  1.136]])
        self.xyz['oh'] = np.array([[ 0.000,  0.000,  0.000],
                                   [-0.913,  0.000,  0.297]])
        self.xyz['al'] = np.array([[ 0.000,  0.000,  0.000],
                                   [-1.011,  0.000,  0.700],
                                   [ 0.998,  0.000,  0.463]])
        self.xyz['nt'] = np.array([[ 0.000,  0.000,  0.000],
                                   [-1.105,  0.000,  0.563],
                                   [ 1.105,  0.000,  0.563]])
        self.xyz['f'] = np.array([[ 0.000,  0.000,  0.000]])
        self.xyz['sh'] = np.array([[ 0.000,  0.000,  0.000],
                                   [-1.331,  0.000,  0.156]])
        self.xyz['sf'] = np.array([[ 0.000,  0.000,  0.000],
                                   [ 0.548, -1.266,  0.448],
                                   [ 0.548,  1.266,  0.448],
                                   [-1.311,  0.000,  0.279]])
        self.xyz['cl'] = np.array([[ 0.000,  0.000,  0.000]])

    def _add_comb(self):
        """Adds substituents made by combining multiple substituents."""
        self.elem['et'], self.xyz['et'] = self.add_subs('me', 'me')
        self.elem['npr'], self.xyz['npr'] = self.add_subs('me', 'me', 'me')
        self.elem['ipr'], self.xyz['ipr'] = self.add_subs('me', 'me', 'me',
                                                          inds=1)
        self.elem['nbu'], self.xyz['nbu'] = self.add_subs('me', 'me', 'me',
                                                          'me')
        self.elem['ibu'], self.xyz['ibu'] = self.add_subs('me', 'me', 'me',
                                                          'me', inds=[2, 1, -1])
        self.elem['tbu'], self.xyz['tbu'] = self.add_subs('me', 'me', 'me',
                                                          'me', inds=1)
        self.elem['ome'], self.xyz['ome'] = self.add_subs('oh', 'me')
        self.elem['ac'], self.xyz['ac'] = self.add_subs('al', 'me')
        self.elem['ca'], self.xyz['ca'] = self.add_subs('al', 'oh')
        self.elem['tfm'], self.xyz['tfm'] = self.add_subs('me', 'f', 'f', 'f',
                                                          inds=1)
        self.elem['ms'], self.xyz['ms'] = self.add_subs('sf', 'me')

    def get_sub(self, label):
        """Returns the element list and cartesian geometry of a
        substituent.

        Parameters
        ----------
        label : str
            The substituent label of the desired substituent.

        Returns
        -------
        elem : (N,) ndarray
            The atomic symbols of the substituent.
        xyz : (N, 3) ndarray
            The atomic cartesian coordinates of the substituent.
        """
        lbl = self.syn[label.lower()]
        return self.elem[lbl], self.xyz[lbl]

    def add_subs(self, *lbls, inds=-1):
        """Returns the element list and cartesian geometry from a
        combination of substituents.

        Parameters
        ----------
        lbls : list
            A list of substituent labels to be combined.
        inds : int or array_like, optional
            The indices for substitution between substituents. Setting
            inds=-1 (default) makes the last atom the subtituted atom.
            Otherwise a list of indices can be given for the first of
            each pair of substituents.

        Returns
        -------
        elem : (N,) ndarray
            The atomic symbols of the combined substituent.
        xyz : (N, 3) ndarray
            The atomic cartesian coordinates of the combined substituent.
        """
        if isinstance(inds, int):
            inds = (len(lbls) - 1) * [inds]
        elif len(inds) != len(lbls) - 1:
            raise ValueError('Number of inds != number of labels - 1')

        rot = 0
        lbl0 = self.syn[lbls[0].lower()]
        elem = self.elem[lbl0]
        xyz = self.xyz[lbl0]
        for i, label in zip(inds, lbls[1:]):
            dist = np.linalg.norm(xyz - xyz[i], axis=1)
            dist[i] += np.max(dist)
            ibond = np.argmin(dist)
            rot = (rot + 1) % 2
            ax = con.unit_vec(xyz[i] - xyz[ibond])
            lbl = self.syn[label.lower()]
            new_elem = self.elem[lbl]
            new_xyz = displace.rotate(self.xyz[lbl], rot*np.pi, 'Z')
            new_xyz = displace.align_axis(new_xyz, 'Z', ax)
            blen = con.get_covrad(elem[ibond]) + con.get_covrad(new_elem[0])
            new_xyz += xyz[ibond] + blen * ax
            elem = np.hstack((np.delete(elem, i), new_elem))
            xyz = np.vstack((np.delete(xyz, i, axis=0), new_xyz))

        return elem, xyz


def import_sub(label):
    """Returns the element list and cartesian geometry of a substituent
    given its label.

    Parameters
    ----------
    label : str
        The substituent label.

    Returns
    -------
    elem : (N,) ndarray
        The atomic symbols of the substituent.
    xyz : (N, 3) ndarray
        The atomic cartesian coordinates of the substituent.
    """
    lib = SubLib()
    return lib.get_sub(label)


def subst(elem, xyz, sublbl, isub, ibond=None, pl=None, vec=None):
    """Returns a molecular geometry with an specified atom replaced by
    substituent.

    Labels are case-insensitive. The index isub gives the position to be
    substituted. If specified, ibond gives the atom bonded to the
    substituent. Otherwise, the nearest atom to isub is used. The
    orientation of the substituent can be given as a vector (the plane
    normal) or an index (the plane containing isub, ibond and pl).

    If isub is given as a list, the entire list of atoms is removed
    and the first index is treated as the position of the substituent.

    Parameters
    ----------
    elem : (N,) array_like
        The atomic symbols of the unsubstituted molecule.
    xyz : (N, 3) array_like
        The atomic cartesian coordinates of the unsubstituted molecule.
    sublbl : str
        The substituent label.
    isub : int or list
        The atomic index (or indices) to be replaced by the substituent.
    ibond : int, optional
        The atomic index of the atom bonded to position isub. If None
        (default), the nearest atom is chosen.
    pl : int or array_like, optional
        The atomic index or vector defining the xz-plane of the
        substituent. If an index is given, the plane normal to the
        isub-ibond-pl plane is used. If None (default), the plane
        is arbitrarily set to [1, 1, 1] and the bond axis is projected
        out.
    vec : (N, 3) array_like, optional
        The atomic cartesian vectors of the unsubstitued molecule. Default
        is None.

    Returns
    -------
    new_elem : (N,) ndarray
        The atomic symbols of the substituted molecule.
    new_xyz : (N, 3) ndarray
        The atomic cartesian coordinates of the substituted molecule.
    new_vec : (N, 3) ndarray
        The atomic cartesian vectors of the substituted molecule.
        Substituent atoms are all set of zero. If vec is None, new_vec
        is all zeros.
    """
    elem = np.array(elem)
    xyz = np.atleast_2d(xyz)
    if not isinstance(isub, int):
        ipos = isub[0]
    else:
        isub = [isub]
        ipos = isub[0]

    if ibond is None:
        dist = np.linalg.norm(xyz - xyz[ipos], axis=1)
        dist[ipos] += np.max(dist)
        ibond = np.argmin(dist)
    elif ibond == ipos:
        raise ValueError('sub and bond indices cannot be the same')

    ax = con.unit_vec(xyz[ipos] - xyz[ibond])
    if pl is None:
        # choose an arbitrary axis and project out the bond axis
        pl = np.ones(3)
        pl -= np.dot(pl, ax) * ax
    elif isinstance(pl, int):
        if pl == ipos:
            raise ValueError('plane and sub indices cannot be the same')
        elif pl == ibond:
            raise ValueError('plane and bond indices cannot be the same')
        pl = np.cross(xyz[ipos] - xyz[ibond], xyz[pl] - xyz[ibond])

    sub_el, sub_xyz = import_sub(sublbl)
    if elem[ipos] == sub_el[0]:
        blen = np.linalg.norm(xyz[ipos] - xyz[ibond])
    else:
        blen = con.get_covrad(elem[ibond]) + con.get_covrad(sub_el[0])

    # rotate to correct orientation and displace to correct position
    sub_xyz = displace.align_axis(sub_xyz, 'Z', ax)
    sub_pl = displace.align_axis([0., 1., 0.], 'Z', ax)
    sub_xyz = displace.align_axis(sub_xyz, sub_pl, pl)
    sub_xyz += xyz[ibond] + blen * ax

    # build the final geometry
    ind1 = [i for i in range(ipos) if i not in isub[1:]]
    ind2 = [i for i in range(ipos+1, len(elem)) if i not in isub[1:]]
    new_elem = np.hstack((elem[ind1], sub_el, elem[ind2]))
    new_xyz = np.vstack((xyz[ind1], sub_xyz, xyz[ind2]))
    if vec is None:
        return new_elem, new_xyz, None
    else:
        new_vec = np.vstack((vec[ind1], np.zeros((len(sub_el), 3)), vec[ind2]))
        return new_elem, new_xyz, new_vec
