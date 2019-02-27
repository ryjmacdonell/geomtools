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
        """Checks that xyz is 3D and len(elem) = len(xyz)."""
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
        """Creates a copy of the BaseMolecule object."""
        self._check()
        if comment is None:
            comment = self._comment
        return BaseMolecule(np.copy(self._elem), np.copy(self._xyz),
                            np.copy(self._vec), comment)

    def save(self):
        """Saves molecular properties to 'save' variables."""
        self._check()
        self.save_elem = np.copy(self._elem)
        self.save_xyz = np.copy(self._xyz)
        self.save_vec = np.copy(self._vec)
        self.save_comment = np.copy(self._comment)
        self.save_print_vec = np.copy(self.print_vec)
        self.saved = True

    def revert(self):
        """Reverts properties to 'save' variables."""
        if not self.saved:
            self._elem = np.copy(self.save_elem)
            self._xyz = np.copy(self.save_xyz)
            self._vec = np.copy(self.save_vec)
            self._comment = np.copy(self.save_comment)
            self.print_vec = np.copy(self.save_print_vec)
        self.saved = True

    def add_atoms(self, new_elem, new_xyz, new_vec=None):
        """Adds atoms(s) to molecule."""
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
        """Removes atom(s) from molecule by index."""
        self._elem = np.delete(self._elem, ind)
        self._xyz = np.delete(self._xyz, ind, axis=0)
        self._vec = np.delete(self._vec, ind, axis=0)
        self._check()
        self.saved = False

    def rearrange(self, new_ind, old_ind=None):
        """Moves atom(s) from old_ind to new_ind."""
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
        atom is at centre of mass."""
        BaseMolecule._check(self, ielem=ielem, ixyz=ixyz, ivec=ivec)
        if self.natm > 0:
            self._add_centre()

    def copy(self, comment=None):
        """Creates a copy of the Molecule object."""
        self._check()
        vec = self._vec if self.print_vec else None
        if comment is None:
            comment = self._comment
        return Molecule(np.copy(self._elem), np.copy(self._xyz),
                        vec, comment)

    def read(self, infile, fmt='auto', **kwargs):
        """Reads single geometry from input file in provided format."""
        read_func = getattr(fileio, 'read_' + fmt)
        if isinstance(infile, str):
            with open(infile, 'r') as f:
                (self._elem, self._xyz,
                 ivec, self._comment) = read_func(f, **kwargs)
        else:
            (self._elem, self._xyz,
             ivec, self._comment) = read_func(infile, **kwargs)

        if ivec is None:
            self._vec = np.zeros_like(self._xyz)
        else:
            self._vec = ivec
        self._check()
        self.saved = False

    def write(self, outfile, fmt='auto', **kwargs):
        """Writes geometry to an output file in provided format."""
        write_func = getattr(fileio, 'write_' + fmt)
        vec = self._vec[1:] if self.print_vec else None
        if isinstance(outfile, str):
            with open(outfile, 'w') as f:
                write_func(f, self._elem[1:], self._xyz[1:], vec=vec,
                           comment=self._comment, **kwargs)
        else:
            write_func(outfile, self._elem[1:], self._xyz[1:], vec=vec,
                       comment=self._comment, **kwargs)

    def get_mass(self):
        """Returns atomic masses."""
        return con.get_mass(self._elem)

    def get_formula(self):
        """Gets the atomic formula based on the element list."""
        elem = [sym for sym in self.elem if 'X' not in sym]
        atm, num = np.unique(elem, return_counts=True)
        fstr = ''
        for a, n in zip(atm, num):
            if n == 1:
                fstr += a
            else:
                fstr += a + str(n)

        return fstr

    def measure(self, coord, *inds, units='auto', absv=False):
        """Returns a coordinate based on its indices in the molecule."""
        self._check()
        coord_func = getattr(measure, coord)
        return coord_func(self.xyz, *inds, units=units, absv=absv)

    def centre_mass(self):
        """Places the centre of mass at the origin."""
        self._check()
        self._xyz -= self._xyz[0]
        self.saved = False

    def translate(self, amp, axis, ind=None, units='ang'):
        """Translates the molecule along a given axis.

        Momenta are difference vectors and are not translated.
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
        """
        kwargs = dict(ind=ind, origin=origin, det=det, units=units)
        self.xyz = displace.rotate(self._xyz, amp, axis, **kwargs)
        if self.print_vec:
            self.vec = displace.rotate(self._vec, amp, axis, **kwargs)
        self.saved = False

    def match_to_ref(self, ref_bundle, plist=None, equiv=None, wgt=None,
                     ind=None, cent=None):
        """Tests the molecule against a set of references in a bundle.

        Note: vectors are not properly rotated.
        """
        vec = self._vec if self.print_vec else None
        reflist = [mol._xyz for mol in ref_bundle.molecules]
        kwargs = dict(plist=plist, equiv=equiv, wgt=wgt, ind=ind, cent=cent)
        self.xyz, ind = kabsch.opt_ref(self._elem, self._xyz, reflist, **kwargs)
        return ind

    def subst(self, lbl, isub, ibond=None, pl=None):
        """Replaces an atom or set of atoms with a substituent."""
        args = (self._elem, self._xyz, lbl, isub)
        kwargs = dict(ibond=ibond, pl=pl, vec=self._vec)
        self._elem, self._xyz, self._vec = substitute.subst(*args, **kwargs)
        self._check()


class MoleculeBundle(object):
    """
    Object containing a set of molecules in the form of Molecule
    objects.
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
        """Check that all bundle objects are Molecule type."""
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
        """Returns a copy of the MoleculeBundle object."""
        return MoleculeBundle([mol.copy() for mol in self._molecules])

    def save(self):
        """Saves all molecules in the bundle."""
        self._check()
        for mol in self._molecules:
            mol.save()
        self.save_molecules = np.copy(self._molecules)
        self.saved = True

    def revert(self):
        """Reverts each molecule in the bundle."""
        if not self.saved:
            self._molecules = np.copy(self.save_molecules)
        for mol in self._molecules:
            mol.revert()
        self.saved = True

    def rearrange(self, new_ind, old_ind=None):
        """Moves molecule(s) from old_ind to new_ind in bundle."""
        if old_ind is None:
            old_ind = range(self.nmol)
        new, old = _rearrange_inds(new_ind, old_ind)
        self._molecules[old] = self._molecules[new]
        self.saved = False

    def add_molecules(self, new_molecules):
        """Adds molecule(s) to the bundle."""
        mols = np.hstack((self._molecules, new_molecules))
        self._check(imol=mols)
        self._molecules = mols
        self.saved = False

    def rm_molecules(self, ind):
        """Removes molecule(s) from the bundle by index."""
        self._molecules = np.delete(self._molecules, ind)
        self._check()
        self.saved = False

    def read(self, infile, fmt='auto', **kwargs):
        """Reads all geometries from input file in provided format."""
        read_func = getattr(fileio, 'read_' + fmt)
        if isinstance(infile, str):
            infile = open(infile, 'r')
        while True:
            try:
                new_mol = Molecule(*read_func(infile, **kwargs))
                self._molecules = np.hstack((self.molecules, new_mol))
            except IOError:
                break

        self._check()
        self.saved = False

    def write(self, outfile, fmt='auto', **kwargs):
        """Writes geometries to an output file in provided format."""
        write_func = getattr(fileio, 'write_' + fmt)
        if isinstance(outfile, str):
            with open(outfile, 'w') as f:
                for mol in self.molecules:
                    mol.write(f, fmt=fmt, **kwargs)
        else:
            for mol in self.molecules:
                mol.write(outfile, fmt=fmt, **kwargs)

    def measure(self, coord, *inds, units='auto', absv=False):
        """Returns a list of coordinates based on index in molecules."""
        kwargs = dict(units=units, absv=absv)
        return np.array([mol.measure(coord, *inds, **kwargs)
                         for mol in self.molecules])

    def match_to_ref(self, ref_bundle, plist=None, equiv=None, wgt=None,
                     ind=None, cent=None):
        """Tests the molecules in the current bundle against
        a set of references in another bundle.

        Returns a set of bundles corresponding to the reference indices.
        """
        kwargs = dict(plist=plist, equiv=equiv, wgt=wgt, ind=ind, cent=cent)
        #bundles = [MoleculeBundle() for mol in ref_bundle.molecules]
        inds = np.empty(self.nmol)
        for i, mol in enumerate(self._molecules):
            inds[i] = mol.match_to_ref(ref_bundle, **kwargs)
        return inds


def import_molecule(fname, fmt='auto', **kwargs):
    """Imports geometry in provided format to Molecule object."""
    read_func = getattr(fileio, 'read_' + fmt)
    with open(fname, 'r') as infile:
        return Molecule(*read_func(infile, **kwargs))


def import_bundle(fnamelist, fmt='auto', **kwargs):
    """Imports geometries in provided format to MoleculeBundle object.

    The fnamelist keyword can be a single filename or a list of
    filenames. If fmt='auto', different files may have different formats.
    """
    read_func = getattr(fileio, 'read_' + fmt)
    molecules = []
    if not isinstance(fnamelist, (list, tuple, np.ndarray)):
        fnamelist = [fnamelist]

    for fname in fnamelist:
        with open(fname, 'r') as infile:
            while True:
                try:
                    molecules.append(Molecule(*read_func(infile, **kwargs)))
                except IOError:
                    break

    return MoleculeBundle(molecules)


def _rearrange_inds(new_ind, old_ind):
    """Checks indices of rearrangement routines and returns indices in
    a single format."""
    new = np.atleast_1d(new_ind)
    old = np.atleast_1d(old_ind)
    if len(new) != len(old):
        raise IndexError('Old and new indices must be the same length')
    return np.hstack((old, new)), np.hstack((new, old))


def _add_type(inp1, inp2):
    """Checks the type of an object being added to a bundle."""
    if isinstance(inp1, MoleculeBundle) and isinstance(inp2, MoleculeBundle):
        molecules = np.hstack((inp1.molecules, inp2.molecules))
    elif isinstance(inp1, MoleculeBundle) and isinstance(inp2, Molecule):
        molecules = np.hstack((inp1.molecules, inp2))
    elif isinstance(inp1, Molecule) and isinstance(inp2, MoleculeBundle):
        molecules = np.hstack((inp1, inp2.molecules))
    elif isinstance(inp1, Molecule) and isinstance(inp2, Molecule):
        molecules = np.hstack((inp1, inp2))
    else:
        raise TypeError('Addition not supported for types \'{:s}\' and '
                        '\'{:s}\'.'.format(type(inp1), type(inp2)))
    return molecules
