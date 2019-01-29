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
import geomtools.fileio as fileio
import geomtools.displace as displace
import geomtools.substitute as substitute
import geomtools.constants as con
import geomtools.kabsch as kabsch


class BaseMolecule(object):
    """
    Basic object containing molecular geometry and functions for setting and
    changing the geometry.

    All methods of BaseMolecule involve setting and saving the molecular
    geometry. There are no dependancies to other geomtools modules.
    """
    def __init__(self, elem=np.array([], dtype=str), xyz=np.empty((0, 3)),
                 mom=None, comment=''):
        self.elem = np.array(elem, dtype=str)
        self.xyz = np.array(xyz, dtype=float)
        self.comment = comment
        self.natm = len(elem)
        if mom is None:
            self.mom = np.zeros((self.natm, 3))
            self.print_mom = False
        else:
            self.mom = np.array(mom, dtype=float)
            self.print_mom = True
        self.saved = True
        self.save()

    def __repr__(self):
        nelem = len(self.elem)
        fmt = '[{:>2s},' + 2*'{:16.8e},' + '{:16.8e}'
        if self.print_mom:
            xyzmom = np.hstack((self.xyz, self.mom))
            fmt += ',\n' + 6*' ' + 2*'{:16.8e},' + '{:16.8e}]'
        else:
            xyzmom = self.xyz
            fmt += ']'

        if nelem == 0:
            fstr = 'BaseMolecule({!r}, ['.format(self.comment)
        else:
            fstr = 'BaseMolecule({!r},\n ['.format(self.comment)

        if nelem > 10:
            fstr += fmt.format(self.elem[0], *xyzmom[0])
            for i in range(1, 3):
                fstr += ',\n  ' + fmt.format(self.elem[i], *xyzmom[i])
            fstr += ',\n  ...'
            for i in range(-3, 0):
                fstr += ',\n  ' + fmt.format(self.elem[i], *xyzmom[i])
        else:
            for i in range(nelem):
                if i != 0:
                    fstr += ',\n  '
                fstr += fmt.format(self.elem[i], *xyzmom[i])

        fstr += '])'
        return fstr

    def __str__(self):
        nelem = len(self.elem)
        fmt = '[{:>2s}' + 3*'{:14.8f}'
        if self.print_mom:
            xyzmom = np.hstack((self.xyz, self.mom))
            fmt += '\n' + 4*' ' + 3*'{:14.8f}' + ']'
        else:
            xyzmom = self.xyz
            fmt += ']'
        fstr = ''
        if self.comment != '':
            fstr += '{!s}\n['.format(self.comment)
        else:
            fstr += '['
        if nelem > 10:
            fstr += fmt.format(self.elem[0], *xyzmom[0])
            for i in range(1, 3):
                fstr += '\n ' + fmt.format(self.elem[i], *xyzmom[i])
            fstr += '\n ...'
            for i in range(-3, 0):
                fstr += '\n ' + fmt.format(self.elem[i], *xyzmom[i])
        else:
            for i in range(nelem):
                if i != 0:
                    fstr += '\n '
                fstr += fmt.format(self.elem[i], *xyzmom[i])
        fstr += ']'
        return fstr

    def _check(self):
        """Checks that xyz is 3D and len(elem) = len(xyz)."""
        if self.xyz.shape[1] != 3:
            raise ValueError('Molecular geometry must be 3-dimensional.')
        if self.mom.shape[1] != 3:
            raise ValueError('Molecular momentum must be 3-dimensional.')
        len_elem = len(self.elem)
        len_xyz = len(self.xyz)
        if len_elem != len_xyz:
            raise ValueError('Number of element labels ({:d}) not equal '
                             'to number of cartesian vectors '
                             '({:d}).'.format(len_elem, len_xyz))
        elif self.xyz.shape != self.mom.shape:
            raise ValueError('Cartesian geometry and momenta must have '
                             'the same number of elements.')

    def copy(self, comment=None):
        """Creates a copy of the BaseMolecule object."""
        self._check()
        if comment is None:
            comment = 'Copy of ' + self.comment
        return BaseMolecule(np.copy(self.elem), np.copy(self.xyz),
                            np.copy(self.mom), comment)

    def save(self):
        """Saves molecular properties to 'orig' variables."""
        self._check()
        self.orig_elem = np.copy(self.elem)
        self.orig_xyz = np.copy(self.xyz)
        self.orig_mom = np.copy(self.mom)
        self.orig_comment = np.copy(self.comment)
        self.saved = True

    def revert(self):
        """Reverts properties to 'orig' variables."""
        if not self.saved:
            self.elem = np.copy(self.orig_elem)
            self.xyz = np.copy(self.orig_xyz)
            self.mom = np.copy(self.orig_mom)
        self.saved = True

    def set_geom(self, elem, xyz):
        """Sets molecular geometry."""
        if elem is not None:
            self.elem = elem
        self.xyz = np.array(xyz, dtype=float)
        self.mom = np.zeros_like(xyz)
        self._check()
        self.saved = False

    def set_mom(self, mom):
        """Sets molecular momentum."""
        self.mom = np.array(mom, dtype=float)
        self.print_mom = True
        self._check()
        self.saved = False

    def set_comment(self, comment):
        """Adds a comment line to describe the molecule."""
        self.comment = comment
        self.saved = False

    def add_atoms(self, new_elem, new_xyz, new_mom=None):
        """Adds atoms(s) to molecule."""
        self.natm += 1 if isinstance(new_elem, str) else len(new_elem)
        new_xyz = np.atleast_2d(new_xyz)
        self.elem = np.hstack((self.elem, new_elem))
        self.xyz = np.vstack((self.xyz, new_xyz))
        if new_mom is None:
            self.mom = np.vstack((self.mom, np.zeros((len(new_xyz), 3))))
        else:
            new_mom = np.atleast_2d(new_mom)
            self.mom = np.vstack((self.mom, new_mom))
            self.print_mom = True
        self._check()
        self.saved = False

    def rm_atoms(self, ind):
        """Removes atom(s) from molecule by index."""
        self.natm -= 1 if isinstance(ind, int) else len(ind)
        self.elem = np.delete(self.elem, ind)
        self.xyz = np.delete(self.xyz, ind, axis=0)
        self.mom = np.delete(self.mom, ind, axis=0)
        self._check()
        self.saved = False

    def rearrange(self, new_ind, old_ind=None):
        """Moves atom(s) from old_ind to new_ind."""
        if old_ind is None:
            old_ind = range(self.natm)
        _rearrange_check(new_ind, old_ind)
        old = np.hstack((new_ind, old_ind))
        new = np.hstack((old_ind, new_ind))
        self.xyz[old] = self.xyz[new]
        self.mom[old] = self.mom[new]
        self.elem[old] = self.elem[new]
        self.saved = False


class Molecule(BaseMolecule):
    """
    More advanced Molecule object which inherits from BaseMolecule.

    A dummy atom XM is added to index 0 at the centre of mass of the
    molecule. Indices thus match the regular definition. Accessor methods
    get_elem and get_xyz return the geometry without the index 0 dummy atom.

    Molecule can also read from and write to input files given by a filename
    or an open file object. Other methods are derived from geomtools modules.
    """
    def __repr__(self):
        baserepr = BaseMolecule.__repr__(self)
        return baserepr.replace('BaseMolecule', 'Molecule', 1)

    def __str__(self):
        base = BaseMolecule(self.elem[1:], self.xyz[1:],
                            self.mom[1:] if self.print_mom else None,
                            self.comment)
        return base.__str__()

    def __add__(self, other):
        return MoleculeBundle(_add_type(self, other))

    def _add_centre(self):
        """Adds a dummy atom at index 0 at molecular centre of mass."""
        pos = displace.get_centremass(self.elem, self.xyz)
        if self.elem[0] == 'XM':
            self.xyz[0] = pos
        else:
            self.elem = np.hstack(('XM', self.elem))
            self.xyz = np.vstack((pos, self.xyz))
            self.mom = np.vstack(([0, 0, 0], self.mom))

    def _check(self):
        """Checks that xyz is 3D and len(elem) = len(xyz) and that dummy
        atom is at centre of mass."""
        BaseMolecule._check(self)
        if self.natm > 0:
            self._add_centre()

    def copy(self, comment=None):
        """Creates a copy of the Molecule object."""
        self._check()
        mom = self.mom[1:] if self.print_mom else None
        if comment is None:
            comment = self.comment
        return Molecule(np.copy(self.elem[1:]), np.copy(self.xyz[1:]),
                        mom, comment)

    def set_mom(self, mom):
        """Sets the molecular momentum."""
        new_mom = np.vstack(([0, 0, 0], mom))
        BaseMolecule.set_mom(self, new_mom)

    # Input/Output
    def read(self, infile, fmt='auto', hasmom=False, hascom=False):
        """Reads single geometry from input file in provided format."""
        read_func = getattr(fileio, 'read_' + fmt)
        if isinstance(infile, str):
            with open(infile, 'r') as f:
                (self.elem, self.xyz,
                 self.mom, self.comment) = read_func(f, hasmom=hasmom,
                                                     hascom=hascom)
        else:
            (self.elem, self.xyz,
             self.mom, self.comment) = read_func(infile, hasmom=hasmom,
                                                 hascom=hascom)
        self.natm = len(self.elem)
        self.save()

    def write(self, outfile, fmt='auto'):
        """Writes geometry to an output file in provided format."""
        write_func = getattr(fileio, 'write_' + fmt)
        mom = self.mom[1:] if self.print_mom else None
        if isinstance(outfile, str):
            with open(outfile, 'w') as f:
                write_func(f, self.elem[1:], self.xyz[1:], mom=mom,
                           comment=self.comment)
        else:
            write_func(outfile, self.elem[1:], self.xyz[1:], mom=mom,
                       comment=self.comment)

    # Accessors
    def get_natm(self):
        """Returns number of atoms."""
        return self.natm

    def get_elem(self):
        """Returns list of elements."""
        return self.elem[1:]

    def get_xyz(self):
        """Returns cartesian geometry."""
        return self.xyz[1:]

    def get_mom(self):
        """Return cartesian momenta."""
        return self.mom[1:]

    def get_comment(self):
        """Returns comment line."""
        return self.comment

    def get_mass(self):
        """Returns atomic masses."""
        return con.get_mass(self.elem[1:])

    def get_form(self):
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

    # Internal geometry
    def get_stre(self, ind, units='ang', absv=False):
        """Returns bond length based on index in molecule."""
        return displace.stre(self.xyz, ind, units=units, absv=absv)

    def get_bend(self, ind, units='rad', absv=False):
        """Returns bond angle based on index in molecule."""
        return displace.bend(self.xyz, ind, units=units, absv=absv)

    def get_tors(self, ind, units='rad', absv=False):
        """Returns dihedral angle based on index in molecule."""
        return displace.tors(self.xyz, ind, units=units, absv=absv)

    def get_oop(self, ind, units='rad', absv=False):
        """Returns out-of-plane angle based on index in molecule."""
        return displace.oop(self.xyz, ind, units=units, absv=absv)

    def get_planeang(self, ind, units='rad', absv=False):
        """Returns plane angle based on index in molecule."""
        return displace.planeang(self.xyz, ind, units=units, absv=absv)

    def get_planetors(self, ind, units='rad', absv=False):
        """Returns plane dihedral angle based on index in molecule."""
        return displace.planetors(self.xyz, ind, units=units, absv=absv)

    def get_edgetors(self, ind, units='rad', absv=False):
        """Returns edge dihedral angle based on index in molecule."""
        return displace.edgetors(self.xyz, ind, units=units, absv=absv)

    # Displacement
    def centre_mass(self):
        """Places the centre of mass at the origin."""
        self.xyz -= self.xyz[0]
        self.saved = False

    def translate(self, amp, axis, ind=None, units='ang'):
        """Translates the molecule along a given axis.

        Momenta are difference vectors and are not translated.
        """
        self.xyz = displace.translate(self.xyz, amp, axis, ind=ind,
                                      units=units)
        self._add_centre()
        self.saved = False

    def rotate(self, amp, axis, ind=None, origin=np.zeros(3), det=1,
               units='rad'):
        """Rotates the molecule about a given axis from a given origin.

        If momenta are non-zero, they will be rotated about the
        same origin. Reflections and improper rotations can be done
        by setting det=-1.
        """
        kwargs = dict(ind=ind, origin=origin, det=det, units=units)
        self.xyz = displace.rotate(self.xyz, amp, axis, **kwargs)
        if self.print_mom:
            self.mom = displace.rotate(self.mom, amp, axis, **kwargs)
        self._add_centre()
        self.saved = False

    # Kabsch geometry matching
    def match_to_ref(self, ref_bundle, plist=None, equiv=None, wgt=None,
                     ind=None, cent=None):
        """Tests the molecule against a set of references in a bundle.

        Note: momenta are not properly rotated.
        """
        mom = self.mom[1:] if self.print_mom else None
        reflist = [mol.get_xyz() for mol in ref_bundle.get_molecules()]
        kwargs = dict(plist=_atm_inds(plist), equiv=_atm_inds(equiv),
                      wgt=wgt, ind=ind, cent=cent)
        xyz, ind = kabsch.opt_ref(self.get_elem(), self.get_xyz(), reflist,
                                  **kwargs)
        return Molecule(self.get_elem(), xyz, mom,
                        self.get_comment()), ind

    # Functional group substitution
    def subst(self, lbl, isub, ibond, iplane=None):
        """Replaces an atom or set of atoms with a substituent."""
        args = (self.elem, self.xyz, lbl, isub, ibond)
        kwargs = dict(iplane=iplane, mom=self.mom)
        self.elem, self.xyz, self.mom = substitute.subst(*args, **kwargs)


class MoleculeBundle(object):
    """
    Object containing a set of molecules in the form of Molecule
    objects.
    """
    def __init__(self, molecules=None):
        if molecules is None:
            self.molecules = np.array([], dtype=object)
        else:
            self.molecules = np.array(molecules, copy=False)
        self.nmol = len(self.molecules)
        self._check()

    def __repr__(self):
        if self.nmol == 0:
            fstr = ''
        else:
            fstr = '\n '
            if self.nmol > 6:
                fstr += self._join_str(',\n\n ', self.molecules[:2], 'r')
                fstr += ',\n\n ...,\n\n '
                fstr += self._join_str(',\n\n ', self.molecules[-2:], 'r')
            else:
                fstr += self._join_str(',\n\n ', self.molecules, 'r')
        return 'MoleculeBundle({:s})'.format(fstr)

    def __str__(self):
        if self.nmol == 0:
            fstr = ''
        elif self.nmol > 6:
            fstr = self._join_str('\n\n ', self.molecules[:2], 's')
            fstr += '\n\n ...\n\n '
            fstr += self._join_str('\n\n ', self.molecules[-2:], 's')
        else:
            fstr = self._join_str('\n\n ', self.molecules, 's')
        return '[{:s}]'.format(fstr)

    def __add__(self, other):
        return MoleculeBundle(_add_type(self, other))

    def __iadd__(self, other):
        return MoleculeBundle(_add_type(self, other))

    def _check(self):
        """Check that all bundle objects are Molecule type."""
        for mol in self.molecules:
            if not isinstance(mol, Molecule):
                raise TypeError('Elements of molecule bundle must be '
                                'Molecule type.')

    def _join_str(self, jn, lst, typ):
        """Returns a string of list elements joined by a separator with
        newline characters added."""
        fmt = '{!' + typ + '}'
        return jn.join(fmt.format(l, t=typ).replace('\n', '\n ')
                       for l in lst)

    def copy(self):
        """Returns a copy of the MoleculeBundle object."""
        return MoleculeBundle([mol.copy() for mol in self.molecules])

    def save(self):
        """Saves all molecules in the bundle."""
        for mol in self.molecules:
            mol.save()
        self._check()

    def revert(self):
        """Reverts each molecule in the bundle."""
        for mol in self.molecules:
            mol.revert()

    def rearrange(self, new_ind, old_ind=None):
        """Moves molecule(s) from old_ind to new_ind in bundle."""
        if old_ind is None:
            old_ind = range(self.nmol)
        _rearrange_check(new_ind, old_ind)
        old = np.hstack((new_ind, old_ind))
        new = np.hstack((old_ind, new_ind))
        self.molecules[old] = self.molecules[new]

    def add_molecules(self, new_molecules):
        """Adds molecule(s) to the bundle."""
        self.nmol += (1 if isinstance(new_molecules, Molecule)
                      else len(new_molecules))
        self.molecules = np.hstack((self.molecules, new_molecules))
        self._check()

    def rm_molecules(self, ind):
        """Removes molecule(s) from the bundle by index."""
        self.nmol -= 1 if isinstance(ind, int) else len(ind)
        self.molecules = np.delete(self.molecules, ind)

    # Input/Output
    def read(self, infile, fmt='auto', hasmom=False, hascom=False):
        """Reads all geometries from input file in provided format."""
        read_func = getattr(fileio, 'read_' + fmt)
        while True:
            try:
                new_mol = Molecule(*read_func(infile, hasmom=hasmom,
                                              hascom=hasccom))
                self.molecules = np.hstack((self.molecules, new_mol))
                self.nmol += 1
            except ValueError:
                break

        self.save()

    def write(self, outfile, fmt='auto'):
        """Writes geometries to an output file in provided format."""
        write_func = getattr(fileio, 'write_' + fmt)
        if isinstance(outfile, str):
            with open(outfile, 'w') as f:
                for mol in self.molecules:
                    mol.write(f, fmt=fmt)
        else:
            for mol in self.molecules:
                mol.write(outfile, fmt=fmt)

    # Accessors
    def get_nmol(self):
        """Returns the number of molecules."""
        return self.nmol

    def get_molecules(self):
        """Returns the list of molecules."""
        return self.molecules

    # Internal coordinates
    def get_coord(self, ctyp, inds, units='auto', absv=False):
        """Returns a list of coordinates based on index in molecule."""
        return np.array([getattr(mol, 'get_'+ctyp)(inds, units=units, absv=absv)
                         for mol in self.molecules])

    # Kabsch geometry matching
    def match_to_ref(self, ref_bundle, plist=None, equiv=None, wgt=None,
                     ind=None, cent=None):
        """Tests the molecules in the current bundle against
        a set of references in another bundle.

        Returns a set of bundles corresponding to the reference indices.
        """
        kwargs = dict(plist=plist, equiv=equiv, wgt=wgt, ind=ind, cent=cent)
        bundles = [MoleculeBundle() for mol in ref_bundle.get_molecules()]
        for mol in self.get_molecules():
            newmol, ind = mol.match_to_ref(ref_bundle, **kwargs)
            bundles[ind].add_molecules(newmol)
        return bundles


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
                except ValueError:
                    break

    return MoleculeBundle(molecules)


def _rearrange_check(new_ind, old_ind):
    """Checks indices of rearrangement routines for errors."""
    new = [new_ind] if isinstance(new_ind, int) else new_ind
    old = [old_ind] if isinstance(old_ind, int) else old_ind
    if len(new) != len(old):
        raise IndexError('Old and new indices must be the same length')


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


def _atm_inds(inds):
    """Substracts one from a list (or list of lists) of indices."""
    if inds is not None:
        if isinstance(inds[0], int):
            return [i - 1 for i in inds]
        else:
            return [[i - 1 for i in sub] for sub in inds]
    else:
        return None
