"""
The Molecule and MoleculeBundle objects and tools for generating and
querying molecular geometries.

Molecule creates a saved copy of the geometry after input for reversion
after an operation. Can add/remove individual atoms or groups or set the
full geometry.

Likewise, MoleculeBundle creates a saved copy of a set molecular
geometries. Input files with multiple geometries can be read to a bundle.
"""
import numpy as np
import gimbal.fileio as fileio
import gimbal.displace as displace
import gimbal.measure as measure
import gimbal.substitute as substitute
import gimbal.constants as con
import gimbal.kabsch as kabsch


class BaseMolecule(object):
    """
    Basic object containing molecular geometry and functions for setting and
    changing the geometry.

    All methods of BaseMolecule involve setting and saving the molecular
    geometry. There are no dependencies to other modules.

    Attributes
    ----------
    elem : (N,) ndarray
        The atomic symbols.
    xyz : (N, 3) ndarray
        The atomic cartesian coordinates.
    vec : (N, 3) ndarray
        The atomic cartesian vectors.
    comment : str
        The comment line.
    natm : int
        The number of atoms N. Automatically set during initialization.
    print_vec : bool
        Specifies if the vectors should be printed/written. Automatically
        set to False if vec is None (default).
    save_elem : (N,) ndarray
        The saved atomic symbols.
    save_xyz : (N, 3) ndarray
        The saved atomic cartesian coordinates.
    save_vec : (N, 3) ndarray
        The saved atomic cartesian vectors.
    save_comment : str
        The saved comment line.
    save_print_vec : bool
        Specifies if the vectors should be printed/written. Automatically
        set to False if vec is None (default).
    saved : bool
        Specifies if the ``save_...`` values are up to date.
    """
    def __init__(self, elem=[], xyz=np.empty((0, 3)), vec=None, comment=''):
        self._elem = np.atleast_1d(np.array(elem, dtype=str))
        self._xyz = np.atleast_2d(np.array(xyz, dtype=float))
        self._comment = repr(str(comment))[1:-1]
        if vec is None:
            self._vec = np.zeros_like(self._xyz)
            self.print_vec = False
        else:
            self._vec = np.atleast_2d(np.array(vec, dtype=float))
            self.print_vec = True
        self.save()

    def __repr__(self):
        nelem = len(self.elem)
        fmt = '[{:>2s},' + 2*'{:16.8e},' + '{:16.8e}'
        if self.print_vec:
            xyzvec = np.hstack((self._xyz, self._vec))
            fmt += ',\n' + 6*' ' + 2*'{:16.8e},' + '{:16.8e}]'
        else:
            xyzvec = self._xyz
            fmt += ']'

        if nelem == 0:
            fstr = 'BaseMolecule({!r}, ['.format(self._comment)
        else:
            fstr = 'BaseMolecule({!r},\n ['.format(self._comment)

        if nelem > 10:
            fstr += fmt.format(self._elem[0], *xyzvec[0])
            for i in range(1, 3):
                fstr += ',\n  ' + fmt.format(self._elem[i], *xyzvec[i])
            fstr += ',\n  ...'
            for i in range(-3, 0):
                fstr += ',\n  ' + fmt.format(self._elem[i], *xyzvec[i])
        else:
            for i in range(nelem):
                if i != 0:
                    fstr += ',\n  '
                fstr += fmt.format(self._elem[i], *xyzvec[i])

        fstr += '])'
        return fstr

    def __str__(self):
        nelem = len(self.elem)
        fmt = '[{:>2s}' + 3*'{:14.8f}'
        if self.print_vec:
            xyzvec = np.hstack((self._xyz, self._vec))
            fmt += '\n' + 4*' ' + 3*'{:14.8f}' + ']'
        else:
            xyzvec = self._xyz
            fmt += ']'
        fstr = ''
        if self.comment != '':
            fstr += '{!s}\n['.format(self.comment)
        else:
            fstr += '['
        if nelem > 10:
            fstr += fmt.format(self._elem[0], *xyzvec[0])
            for i in range(1, 3):
                fstr += '\n ' + fmt.format(self._elem[i], *xyzvec[i])
            fstr += '\n ...'
            for i in range(-3, 0):
                fstr += '\n ' + fmt.format(self._elem[i], *xyzvec[i])
        else:
            for i in range(nelem):
                if i != 0:
                    fstr += '\n '
                fstr += fmt.format(self._elem[i], *xyzvec[i])
        fstr += ']'
        return fstr

    def _check(self, ielem=None, ixyz=None, ivec=None):
        """Checks that xyz is 3D and len(elem) = len(xyz).

        Parameters
        ----------
        ielem : (N,) array_like, optional
            The new atomic symbols to be checked. If None (default), the
            values of self._elem are checked instead.
        ixyz : (N, 3) array_like, optional
            The new atomic cartesian coordiantes to be checked. If None
            (default), the values of self._xyz are checked instead.
        ivec : (N, 3) array_like, optional
            The new atomic cartesian vectors to be checked. If None
            (default), the values of self._vec are checked instead.

        Raises
        ------
        ValueError
            When the shapes of ixyz, elem and vec don't match or ixyz
            and ivec aren't 3D.
        """
        if ielem is None:
            ielem = self._elem
        if ixyz is None:
            ixyz = self._xyz
        if ivec is None:
            ivec = self._vec
        if ixyz.shape[1] != 3:
            raise ValueError('Molecular geometry must be 3-dimensional.')
        if ivec.shape[1] != 3:
            raise ValueError('Molecular vector must be 3-dimensional.')
        natm = len(ielem)
        if natm != len(ixyz):
            raise ValueError('Number of element labels not equal '
                             'to number of cartesian vectors.')
        elif ixyz.shape != ivec.shape:
            if self.print_vec:
                raise ValueError('Cartesian geometry and vector must have '
                                 'the same number of elements.')
            else:
                self._vec = np.zeros_like(ixyz)
        self.natm = natm

    @property
    def elem(self):
        """Gets the value of elem."""
        self._check()
        return self._elem

    @elem.setter
    def elem(self, val):
        """Sets the value of elem."""
        val = np.atleast_1d(np.array(val, dtype=str))
        self._check(ielem=val)
        self._elem = val
        self.saved = False

    @property
    def xyz(self):
        """Gets the value of xyz."""
        self._check()
        return self._xyz

    @xyz.setter
    def xyz(self, val):
        """Sets the value of xyz."""
        val = np.atleast_2d(np.array(val, dtype=float))
        self._check(ixyz=val)
        self._xyz = val
        self.saved = False

    @property
    def vec(self):
        """Gets the value of vec."""
        self._check()
        return self._vec

    @vec.setter
    def vec(self, val):
        """Sets the value of vec."""
        val = np.atleast_2d(np.array(val, dtype=float))
        self._check(ivec=val)
        self._vec = val
        self.print_vec = True
        self.saved = False

    @property
    def comment(self):
        """Gets the value of comment."""
        return self._comment

    @comment.setter
    def comment(self, val):
        """Sets the value of comment."""
        self._comment = repr(str(val))[1:-1]
        self.saved = False

    def copy(self, comment=None):
        """Creates a copy of the BaseMolecule object.

        Parameters
        ----------
        comment : str
            A comment line for the copied BaseMolecule. If None (default),
            self._comment is used.

        Returns
        -------
        BaseMolecule
            A copy of the BaseMolecule instance.
        """
        self._check()
        if comment is None:
            comment = self._comment
        return BaseMolecule(np.copy(self._elem), np.copy(self._xyz),
                            np.copy(self._vec), comment)

    def save(self):
        """Saves molecular properties to ``save_`` variables."""
        self._check()
        self.save_elem = np.copy(self._elem)
        self.save_xyz = np.copy(self._xyz)
        self.save_vec = np.copy(self._vec)
        self.save_comment = np.copy(self._comment)
        self.save_print_vec = np.copy(self.print_vec)
        self.saved = True

    def revert(self):
        """Reverts properties to ``save_`` variables."""
        if not self.saved:
            self._elem = np.copy(self.save_elem)
            self._xyz = np.copy(self.save_xyz)
            self._vec = np.copy(self.save_vec)
            self._comment = np.copy(self.save_comment)
            self.print_vec = np.copy(self.save_print_vec)
        self.saved = True

    def add_atoms(self, new_elem, new_xyz, new_vec=None):
        """Adds atoms(s) to molecule.

        Parameters
        ----------
        new_elem : (n,) array_like
            The new atomic symbols to be added to the current symbol
            list.
        new_xyz : (n, 3) array_like
            The new atomic cartesian coordinates to be added to the
            current coordinate list.
        new_vec : (n, 3) array_like, optional
            The new atomic cartesian vectors to be added to the current
            vector list. If None (default), zeros are added to match
            new_xyz.
        """
        new_xyz = np.atleast_2d(np.array(new_xyz, dtype=float))
        elems = np.hstack((self._elem, new_elem))
        xyzs = np.vstack((self._xyz, new_xyz))
        if new_vec is not None:
            new_vec = np.atleast_2d(np.array(new_vec, dtype=float))
            vecs = np.vstack((self._vec, new_vec))
        else:
            vecs = None
        self._check(ielem=elems, ixyz=xyzs, ivec=vecs)
        self._elem = elems
        self._xyz = xyzs
        if new_vec is not None:
            self._vec = vecs
            self.print_vec = True
        self.saved = False

    def rm_atoms(self, ind):
        """Removes atom(s) from molecule by index.

        Parameters
        ----------
        ind : int or array_like
            Index or indices of atoms to be removed.
        """
        self._elem = np.delete(self._elem, ind)
        self._xyz = np.delete(self._xyz, ind, axis=0)
        self._vec = np.delete(self._vec, ind, axis=0)
        self._check()
        self.saved = False

    def rearrange(self, new_ind, old_ind=None):
        """Moves atom(s) from old_ind to new_ind.

        Parameters
        ----------
        new_ind : array_like
            New indices for the atoms.
        old_ind : array_like, optional
            Old indices for the atoms. If None (default), new_ind must
            include all atoms.
        """
        if old_ind is None:
            old_ind = range(self.natm)
        new, old = _rearrange_inds(new_ind, old_ind)
        self._xyz[old] = self._xyz[new]
        self._vec[old] = self._vec[new]
        self._elem[old] = self._elem[new]
        self._check()
        self.saved = False


class Molecule(BaseMolecule):
    """
    More advanced Molecule object which inherits from BaseMolecule.

    A dummy atom XM is added to index 0 at the centre of mass of the
    molecule. Indices thus match the regular definition. Accessor methods
    get_elem and get_xyz return the geometry without the index 0 dummy atom.

    Molecule can also read from and write to input files given by a filename
    or an open file object. Other methods are derived from different modules.

    Attributes
    ----------
    elem : (N+1,) ndarray
        The atomic symbols.
    xyz : (N+1, 3) ndarray
        The atomic cartesian coordinates.
    vec : (N+1, 3) ndarray
        The atomic cartesian vectors.
    comment : str
        The comment line.
    natm : int
        The number of atoms N. Automatically set during initialization.
    print_vec : bool
        Specifies if the vectors should be printed/written. Automatically
        set to False if vec is None (default).
    """
    def __repr__(self):
        baserepr = BaseMolecule.__repr__(self)
        return baserepr.replace('BaseMolecule', 'Molecule', 1)

    def __str__(self):
        base = BaseMolecule(self.elem[1:], self.xyz[1:],
                            self.vec[1:] if self.print_vec else None,
                            self.comment)
        return base.__str__()

    def __add__(self, other):
        return MoleculeBundle(_add_type(self, other))

    def _add_centre(self):
        """Adds a dummy atom at index 0 at molecular centre of mass."""
        pos = displace.get_centremass(self._elem, self._xyz)
        if self._elem[0] == 'XM':
            self._xyz[0] = pos
        else:
            self._elem = np.hstack(('XM', self._elem))
            self._xyz = np.vstack((pos, self._xyz))
            self._vec = np.vstack(([0, 0, 0], self._vec))

    def _check(self, ielem=None, ixyz=None, ivec=None):
        """Checks that xyz is 3D and len(elem) = len(xyz) and that dummy
        atom is at centre of mass.

        Parameters
        ----------
        ielem : (N,) array_like, optional
            The new atomic symbols to be checked. If None (default), the
            values of self._elem are checked instead.
        ixyz : (N, 3) array_like, optional
            The new atomic cartesian coordiantes to be checked. If None
            (default), the values of self._xyz are checked instead.
        ivec : (N, 3) array_like, optional
            The new atomic cartesian vectors to be checked. If None
            (default), the values of self._vec are checked instead.
        """
        BaseMolecule._check(self, ielem=ielem, ixyz=ixyz, ivec=ivec)
        if self.natm > 0:
            self._add_centre()
            self.natm = len(self._elem) - 1

    def copy(self, comment=None):
        """Creates a copy of the Molecule object.

        Parameters
        ----------
        comment : str
            A comment line for the copied Molecule. If None (default),
            self._comment is used.

        Returns
        -------
        Molecule
            A copy of the Molecule instance.
        """
        self._check()
        vec = self._vec if self.print_vec else None
        if comment is None:
            comment = self._comment
        return Molecule(np.copy(self._elem), np.copy(self._xyz),
                        vec, comment)

    def rearrange(self, new_ind, old_ind=None):
        """Moves atom(s) from old_ind to new_ind.

        Parameters
        ----------
        new_ind : array_like
            New indices for the atoms.
        old_ind : array_like, optional
            Old indices for the atoms. If None (default), new_ind must
            include all atoms except for the dummy atom (index 0).
        """
        if old_ind is None:
            old_ind = range(1, self.natm+1)
        BaseMolecule.rearrange(self, new_ind, old_ind)

    def read(self, infile, fmt='auto', **kwargs):
        """Reads single geometry from input file in provided format.

        Parameters
        ----------
        infile : file or str
            The open input file or filename.
        fmt : str, optional
            The file format. Default is auto (i.e. :func:`fileio.read_auto`).
        kwargs : dict, optional
            Additional keyword arguments for the read function.
        """
        (self._elem, self._xyz, ivec,
         self._comment) = fileio.read_single(infile, fmt=fmt, **kwargs)
        if ivec is None:
            self._vec = np.zeros_like(self._xyz)
        else:
            self._vec = ivec

        self._check()
        self.saved = False

    def write(self, outfile, fmt='auto', **kwargs):
        """Writes geometry to an output file in provided format.

        Parameters
        ----------
        outfile : file or str
            The open output file or filename.
        fmt : str, optional
            The file format. Default is auto (i.e. :func:`fileio.write_auto`).
        kwargs : dict, optional
            Additional keyword arguments for the write function.
        """
        vec = self._vec[1:] if self.print_vec else None
        moldat = (self._elem[1:], self._xyz[1:], vec, self._comment)
        fileio.write_single(outfile, moldat, fmt=fmt, **kwargs)

    def get_mass(self):
        """Returns atomic masses.

        Returns
        -------
        ndarray
            The array of atomic masses.
        """
        return con.get_mass(self._elem)

    def get_formula(self):
        """Gets the atomic formula based on the element list.

        Returns
        -------
        str
            The molecular atomic formula as a string.
        """
        elem = [sym for sym in self.elem if 'X' not in sym]
        atm, num = np.unique(elem, return_counts=True)
        fstr = ''
        for a, n in zip(atm, num):
            if n == 1:
                fstr += a
            else:
                fstr += a + str(n)

        return fstr

    def measure(self, coord, *inds, **kwargs):
        """Returns a coordinate based on its indices in the molecule.

        Parameters
        ----------
        coord : str
            The coordinate specification, given by function names
            in :mod:`measure`.
        inds : list
            The indices used in the coordinate function.
        kwargs : dict, optional
            Additional keyword arguments for the measure functions.

        Returns
        -------
        float
            The value of the specified coordinate.
        """
        self._check()
        coord_func = getattr(measure, coord)
        return coord_func(self.xyz, *inds, **kwargs)

    def centre_mass(self):
        """Places the centre of mass at the origin."""
        self._check()
        self._xyz -= self._xyz[0]
        self.saved = False

    def translate(self, amp, axis, ind=None, units='ang'):
        """Translates the molecule along a given axis.

        Momenta are difference vectors and are not translated.

        Parameters
        ----------
        amp : float
            The distance for translation.
        axis : array_like or str
            The axis of translation, parsed by :func:`displace._parse_axis`.
        ind : array_like, optional
            List of atomic indices to specify which atoms are displaced.
            If ind is None (default) then all atoms are displaced.
        units : str, optional
            The units of length for displacement. Default is angstroms.
        """
        self.xyz = displace.translate(self._xyz, amp, axis, ind=ind,
                                      units=units)
        self.saved = False

    def rotate(self, amp, axis, ind=None, origin=np.zeros(3), det=1,
               units='rad'):
        """Rotates the molecule about a given axis from a given origin.

        If vectors are non-zero, they will be rotated about the
        same origin. Reflections and improper rotations can be done
        by setting det=-1.

        Parameters
        ----------
        amp : float
            The angle of rotation.
        axis : array_like or str
            The axis of rotation, parsed by :func:`displace._parse_axis`.
        ind : array_like, optional
            List of atomic indices to specify which atoms are displaced.
            If ind is None (default) then all atoms are displaced.
        origin : (3,) array_like, optional
            The origin of rotation. Default is the cartesian origin.
        det : float, optional
            The determinant of the rotation. 1 (default) is a proper rotation
            and -1 is an improper rotation (rotation + reflection).
        units : str, optional
            The units of length for displacement. Default is angstroms.
        """
        kwargs = dict(ind=ind, origin=origin, det=det, units=units)
        self.xyz = displace.rotate(self._xyz, amp, axis, **kwargs)
        if self.print_vec:
            self.vec = displace.rotate(self._vec, amp, axis, **kwargs)
        self.saved = False

    def match_to_ref(self, ref_bundle, **kwargs):
        """Tests the molecule against a set of references in a bundle.

        At the moment, vectors are not properly rotated.

        Parameters
        ----------
        ref_bundle : MoleculeBundle
            A molecule bundle object containing the set of reference
            geometries.
        kwargs : dict
            Additional keyword arguments used in :func:`kabsch.opt_permute`.

        Returns
        -------
        int
            The index of the optimal ref geometry.
        """
        vec = self._vec if self.print_vec else None
        reflist = [mol._xyz for mol in ref_bundle.molecules]
        self.xyz, ind = kabsch.opt_ref(self._elem, self._xyz, reflist, **kwargs)
        return ind

    def subst(self, lbl, isub, ibond=None, pl=None):
        """Replaces an atom or set of atoms with a substituent.

        Parameters
        ----------
        lbl : str
            The substituent label.
        isub : int or list
            The atomic index (or indices) to be replaced by the substituent.
        ibond : int, optional
            The atomic index of the atom bonded to position isub. If None
            (default), the nearest atom is chosen.
        pl : int or array_like, optional
            The atomic index or vector defining the xz-plane of the
            substituent. See :func:`substitute.subst` for more details.
        """
        args = (self._elem, self._xyz, lbl, isub)
        kwargs = dict(ibond=ibond, pl=pl, vec=self._vec)
        self._elem, self._xyz, vec = substitute.subst(*args, **kwargs)
        if self.print_vec:
            self._vec = vec
        else:
            self._vec = np.zeros_like(self._xyz)
        self._check()


class MoleculeBundle(object):
    """
    Object containing a set of molecules in the form of Molecule
    objects.

    Attributes
    ----------
    molecules : ndarray
        The array of Molecule objects.
    nmol : int
        The number of molecules, set automatically during initialization.
    save_molecules : ndarray
        The saved array of Molecule objects.
    saved : bool
        Specifies if the save_molecules value is up to date.
    """
    def __init__(self, molecules=[]):
        self._molecules = np.atleast_1d(np.array(molecules, dtype=object,
                                                 copy=False))
        self._check()
        self.save()

    def __repr__(self):
        nmol = len(self.molecules)
        if nmol == 0:
            fstr = ''
        else:
            fstr = '\n '
            if nmol > 6:
                fstr += self._join_str(',\n\n ', self._molecules[:2], 'r')
                fstr += ',\n\n ...,\n\n '
                fstr += self._join_str(',\n\n ', self._molecules[-2:], 'r')
            else:
                fstr += self._join_str(',\n\n ', self._molecules, 'r')
        return 'MoleculeBundle({:s})'.format(fstr)

    def __str__(self):
        nmol = len(self.molecules)
        if nmol == 0:
            fstr = ''
        elif nmol > 6:
            fstr = self._join_str('\n\n ', self._molecules[:2], 's')
            fstr += '\n\n ...\n\n '
            fstr += self._join_str('\n\n ', self._molecules[-2:], 's')
        else:
            fstr = self._join_str('\n\n ', self._molecules, 's')
        return '[{:s}]'.format(fstr)

    @staticmethod
    def _join_str(jn, lst, typ):
        """Returns a string of list elements joined by a separator with
        newline characters added."""
        fmt = '{!' + typ + '}'
        return jn.join(fmt.format(l, t=typ).replace('\n', '\n ')
                       for l in lst)

    def __add__(self, other):
        return MoleculeBundle(_add_type(self, other))

    def __iadd__(self, other):
        return MoleculeBundle(_add_type(self, other))

    def _check(self, imol=None):
        """Check that all bundle objects are Molecule type.

        Parameters
        ----------
        imol : ndarray, optional
            The new array of Molecule objects to be checked. If None
            (default), self._molecules is checked instead.

        Raises
        ------
        TypeError
            When the type of elements of imol aren't Molecule.
        """
        if imol is None:
            imol = self._molecules
        for mol in imol:
            if not isinstance(mol, Molecule):
                raise TypeError('Elements of molecule bundle must be '
                                'Molecule type.')
        self.nmol = len(self._molecules)

    @property
    def molecules(self):
        """Gets the value of molecules."""
        self._check()
        return self._molecules

    @molecules.setter
    def molecules(self, val):
        """Sets the value of molecules."""
        val = np.atleast_1d(np.array(val, dtype=object, copy=False))
        self._check(imol=val)
        self._molecules = val
        self.saved = False

    def copy(self):
        """Returns a copy of the MoleculeBundle object.

        Returns
        -------
        MoleculeBundle
            A copy of the MoleculeBundle instance.
        """
        return MoleculeBundle([mol.copy() for mol in self._molecules])

    def save(self):
        """Saves all molecules in the bundle and sets save_molecules."""
        self._check()
        for mol in self._molecules:
            mol.save()
        self.save_molecules = np.copy(self._molecules)
        self.saved = True

    def revert(self):
        """Reverts each molecule in the bundle to the saved variables."""
        if not self.saved:
            self._molecules = np.copy(self.save_molecules)
        for mol in self._molecules:
            mol.revert()
        self.saved = True

    def rearrange(self, new_ind, old_ind=None):
        """Moves molecule(s) from old_ind to new_ind in bundle.

        Parameters
        ----------
        new_ind : array_like
            New indices for the molecules.
        old_ind : array_like, optional
            Old indices for the molecules. If None (default), new_ind must
            include all molecules.
        """
        if old_ind is None:
            old_ind = range(self.nmol)
        new, old = _rearrange_inds(new_ind, old_ind)
        self._molecules[old] = self._molecules[new]
        self.saved = False

    def add_molecules(self, new_molecules):
        """Adds molecule(s) to the bundle.

        Parameters
        ----------
        new_molecules : (n,) array_like
            The new Molecule objects to be added to the current molecule
            list.
        """
        mols = np.hstack((self._molecules, new_molecules))
        self._check(imol=mols)
        self._molecules = mols
        self.saved = False

    def rm_molecules(self, ind):
        """Removes molecule(s) from the bundle by index.

        Parameters
        ----------
        ind : int or array_like
            Index or indices of molecules to be removed.
        """
        self._molecules = np.delete(self._molecules, ind)
        self._check()
        self.saved = False

    def read(self, inflist, fmt='auto', **kwargs):
        """Reads all geometries from input file in provided format.

        Parameters
        ----------
        infile : array_like
            The open input files or filenames.
        fmt : str, optional
            The file format. Default is auto (i.e. :func:`fileio.read_auto`).
        kwargs : dict, optional
            Additional keyword arguments for the read function.
        """
        moldat = fileio.read_multiple(inflist, fmt=fmt, **kwargs)
        new_mol = [Molecule(*dat) for dat in moldat]
        self._molecules = np.hstack((self.molecules, new_mol))
        self._check()
        self.saved = False

    def write(self, outfile, fmt='auto', **kwargs):
        """Writes geometries to an output file in provided format.

        Parameters
        ----------
        outfile : file or str
            The open output file or filename.
        fmt : str, optional
            The file format. Default is auto (i.e. :func:`fileio.write_auto`).
        kwargs : dict, optional
            Additional keyword arguments for the write function.
        """
        moldat = np.empty(self.nmol, dtype=object)
        for i, mol in enumerate(self.molecules):
            if mol.print_vec:
                moldat[i] = (mol.elem[1:], mol.xyz[1:], mol.vec[1:],
                             mol.comment)
            else:
                moldat[i] = (mol.elem[1:], mol.xyz[1:], None, mol.comment)

        fileio.write_multiple(outfile, moldat, fmt=fmt, **kwargs)

    def measure(self, coord, *inds, **kwargs):
        """Returns a list of coordinates based on index in molecules.

        Parameters
        ----------
        coord : str
            The coordinate specification, given by function names
            in :mod:`measure`.
        inds : list
            The indices used in the coordinate function.
        kwargs : dict, optional
            Additional keyword arguments for the measure functions.

        Returns
        -------
        ndarray
            The values of the specified coordinate for each molecule.
        """
        return np.array([mol.measure(coord, *inds, **kwargs)
                         for mol in self.molecules])

    def match_to_ref(self, ref_bundle, plist=None, equiv=None, wgt=None,
                     ind=None, cent=None):
        """Tests the molecules in the current bundle against
        a set of references in another bundle.

        Returns a set of bundles corresponding to the reference indices.

        Parameters
        ----------
        ref_bundle : MoleculeBundle
            A molecule bundle object containing the set of reference
            geometries.
        kwargs : dict
            Additional keyword arguments used in :func:`kabsch.opt_permute`.

        Returns
        -------
        ndarray
            The indices of the optimal ref geometries for each molecule.
        """
        kwargs = dict(plist=plist, equiv=equiv, wgt=wgt, ind=ind, cent=cent)
        inds = np.empty(self.nmol)
        for i, mol in enumerate(self._molecules):
            inds[i] = mol.match_to_ref(ref_bundle, **kwargs)
        return inds


def import_molecule(fname, fmt='auto', **kwargs):
    """Imports geometry in provided format to Molecule object.

    Parameters
    ----------
    fname : file or str
        The open input file or filename.
    fmt : str, optional
        The file format. Default is auto (i.e. :func:`fileio.read_auto`).
    kwargs : dict, optional
        Additional keyword arguments for the read function.

    Returns
    -------
    Molecule
        The imported Molecule object.
    """
    moldat = fileio.read_multiple(fname, fmt=fmt, **kwargs)
    return Molecule(*moldat[0])


def import_bundle(fnamelist, fmt='auto', **kwargs):
    """Imports geometries in provided format to MoleculeBundle object.

    The fnamelist keyword can be a single filename or a list of
    filenames. If fmt='auto', different files may have different formats.

    Parameters
    ----------
    fnamelist : array_like
        The open input files or filenames.
    fmt : str, optional
        The file format. Default is auto (i.e. :func:`fileio.read_auto`).
    kwargs : dict, optional
        Additional keyword arguments for the read function.

    Returns
    -------
    MoleculeBundle
        The imported MoleculeBundle object.
    """
    moldat = fileio.read_multiple(fnamelist, fmt=fmt, **kwargs)
    molecules = [Molecule(*dat) for dat in moldat]
    return MoleculeBundle(molecules)


def _rearrange_inds(new_ind, old_ind):
    """Checks indices of rearrangement routines and returns indices in
    a single format.

    Parameters
    ----------
    new_ind : array_like
            New indices for the elements.
    old_ind : array_like
            Old indices for the elements.

    Returns
    -------
    new : ndarray
        An array of the new ordering of both old an new indices.
    old : ndarray
        An array of the old ordering of both old an new indices.

    Raises
    ------
    IndexError
        When the shape of new_ind and old_ind don't match.
    """
    new = np.atleast_1d(new_ind)
    old = np.atleast_1d(old_ind)
    if len(new) != len(old):
        raise IndexError('Old and new indices must be the same length')
    return np.hstack((old, new)), np.hstack((new, old))


def _add_type(inp1, inp2):
    """Checks the type of an object being added to a bundle.

    Parameters
    ----------
    inp1 : Molecule or MoleculeBundle
        The first input for concatenation into a list of molecules.
    inp2 : Molecule or MoleculeBundle
        The second input for concatenation into a list of molecules.

    Returns
    -------
    ndarray
        An array of Molecule objects from inp1 and inp2.

    Raises
    ------
    TypeError
        When the different types can't be added to make a MoleculeBundle
        object.
    """
    if isinstance(inp1, MoleculeBundle) and isinstance(inp2, MoleculeBundle):
        molecules = np.hstack((inp1.molecules, inp2.molecules))
    elif isinstance(inp1, MoleculeBundle) and isinstance(inp2, Molecule):
        molecules = np.hstack((inp1.molecules, inp2))
    elif isinstance(inp1, Molecule) and isinstance(inp2, MoleculeBundle):
        molecules = np.hstack((inp1, inp2.molecules))
    elif isinstance(inp1, Molecule) and isinstance(inp2, Molecule):
        molecules = np.hstack((inp1, inp2))
    else:
        fmt = 'Addition not supported for types \'{:s}\' and \'{:s}\'.'
        raise TypeError(fmt.format(type(inp1).__class__.__name__,
                                   type(inp2).__class__.__name__))
    return molecules
