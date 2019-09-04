r"""
Routines for displacing a molecular geometry by translation or
proper/improper rotation.
::

        1
        |
        4
       / \
      2   3

Example axes for displacements:

    1. X1X4 stretch: :math:`r_{14}`
    2. X1X4 torsion: :math:`r_{14}` (for motion of 2, 3)
    3. X1X4X2 bend: :math:`r_{14} \times r_{24}`
    4. X1 out-of-plane: :math:`r_{24} - r_{34}` or
       :math:`(r_{24} \times r_{34}) \times r_{14}`

Each internal coordinate measurement has the option of changing the units
(see the constants module) or taking the absolute value.
"""
import operator as op
import pyparsing as pp
import numpy as np
import gimbal.constants as con


class VectorParser(object):
    """An object for defining and evaluating vector operations on
    a cartesian geometry.

    A new VectorParser instance takes a cartesian geometry as an
    optional input. The instance can be called with a vector (no
    action), 3x3 array (cross product) or string, parsed according
    to the syntax in :func:`~VectorParser.generate_parser`.

    Attributes
    ----------
    xyz : (N, 3) array_like, optional
        The cartesian geometry which defines the indices in parsed
        expressions. If None, only expressions without indices can
        be parsed.
    unop : dict
        A dictionary which defines unary operations.
    bnadd : dict
        A dictionary which defines binary addition operations.
    bnmul : dict
        A dictionary which defines binary multiplication operations.
    bnop : dict
        A dictionary which defines all binary operations.
    axes : dict
        A dictionary which defines cartesian axis labels.
    expr : pyparsing.Forward
        A pyparsing grammar used to evaluate expressions. Automatically
        generated when xyz is set.
    """
    def __init__(self, xyz=None):
        self.unop = {'+': op.pos, '-': op.neg}
        self.bnadd = {'+': op.add, '-': op.sub}
        self.bnmul = {'*': op.mul, '/': op.truediv, 'o': np.dot, 'x': np.cross}
        self.bnop = self.bnadd.copy()
        self.bnop.update(self.bnmul)
        self.axes = dict(X = np.array([1., 0., 0.]),
                         Y = np.array([0., 1., 0.]),
                         Z = np.array([0., 0., 1.]))
        self.xyz = xyz

    def __call__(self, inp, unit=False):
        """Evaluates an expression based on a string.

        Parameters
        ----------
        inp : str or array_like
            A string or array used to specify an axis.
        unit : bool, optional
            Specifies if the axis is converted to a unit vector.

        Returns
        -------
        float or ndarray
            The result of the vector operation.

        Raises
        ------
        ValueError
            If input is not a string, 3-vector or 3x3 array.
        """
        if isinstance(inp, str):
            u = self.expr.parseString(inp, parseAll=True)[0]
        elif len(inp) == 3:
            u = np.array(inp, dtype=float)
            if u.size == 9:
                u = np.cross(u[0] - u[1], u[2] - u[1])
        else:
            raise ValueError('Axis specification not recognized')

        if unit:
            return con.unit_vec(u)
        else:
            return u

    @property
    def xyz(self):
        """Gets the value of xyz."""
        return self._xyz

    @xyz.setter
    def xyz(self, val):
        """Sets the value of xyz and generates the parser."""
        self.expr = self.generate_parser(val)
        self._xyz = val

    def _eval_unary(self, tokens):
        """Evaluates unary operations.

        Parameters
        ----------
        tokens : list
            A list of pyparsing tokens from a matching unary expression.

        Returns
        -------
        float or ndarray
            The expression after unary operation.
        """
        vals = tokens[0]
        return self.unop[vals[0]](vals[1])

    def _eval_binary(self, tokens):
        """Evaluates binary operations.

        Parameters
        ----------
        tokens : list
            A list of pyparsing tokens from a matching binary expression.

        Returns
        -------
        float or ndarray
            The expression after binary operation.
        """
        vals = tokens[0]
        newval = vals[0]
        it = iter(vals[1:])
        for oper in it:
            newval = self.bnop[oper](newval, next(it))

        return newval

    def _eval_power(self, tokens):
        """Evaluates power operations.

        Parameters
        ----------
        tokens : list
            A list of pyparsing tokens from a matching power expression.

        Returns
        -------
        float or ndarray
            The expression after power operation.
        """
        vals = tokens[0]
        newval = vals[-1]
        for v in vals[-3::-2]:
            newval = v**newval

        return newval

    def generate_parser(self, xyz=None):
        """Creates the pyparsing expression based on geometry.

        The syntax is as follows:

            - ``i+`` are indices of xyz and return vectors.
            - ``i+.j`` are floating point numbers (j optional).
            - ``i[j]`` is the j-th (scalar) element of xyz[i].
            - ``X, Y, Z`` are unit vectors along x, y and z axes (uppercase only).
            - ``+`` and ``-`` are addition/subtraction of vectors or scalars.
            - ``*`` and ``/`` are multiplication/division of vectors and scalars
              (elementwise).
            - ``o`` and ``x`` are scalar/vector products of vectors only.
            - ``^`` is the power of a vector/scalar by a scalar (elementwise).
            - ``(`` and ``)`` specify order of operation.
            - ``[i, j, k]`` gives a vector with scalar elements i, j and k.

        Parameters
        ----------
        xyz : (N, 3), array_like, optional
            The cartesian geometry used in index expressions. If not
            provided, strings containing indices will raise an error.

        Returns
        -------
        pyparsing.Forward
            A pyparsing grammar definition.
        """
        expr = pp.Forward()

        # operand types: int, int with index, float, axis or delimited list
        intnum = pp.Word(pp.nums)
        fltind = pp.Word(pp.nums) + '[' + pp.Word(pp.nums) + ']'
        fltnum = pp.Combine(pp.Word(pp.nums) + '.' + pp.Optional(pp.Word(pp.nums)))
        alphax = pp.oneOf(' '.join(self.axes))
        dllist = pp.Suppress('[') + pp.delimitedList(expr) + pp.Suppress(']')
        intnum.setParseAction(lambda t: xyz[int(t[0])])
        fltind.setParseAction(lambda t: xyz[int(t[0])][int(t[2])])
        fltnum.setParseAction(lambda t: float(t[0]))
        alphax.setParseAction(lambda t: self.axes[t[0]])
        dllist.setParseAction(lambda t: np.array(t[:]))
        operand = dllist | alphax | fltnum | fltind | intnum

        # operators: unary, power, binary multiplication/division,
        #    binary addition/subtraction
        sgnop = pp.oneOf(' '.join(self.unop))
        expop = pp.Literal('^')
        mulop = pp.oneOf(' '.join(self.bnmul))
        addop = pp.oneOf(' '.join(self.bnadd))

        # set operator precedence
        prec = [(sgnop, 1, pp.opAssoc.RIGHT, self._eval_unary),
                (expop, 2, pp.opAssoc.LEFT, self._eval_power),
                (mulop, 2, pp.opAssoc.LEFT, self._eval_binary),
                (addop, 2, pp.opAssoc.LEFT, self._eval_binary)]
        return expr << pp.infixNotation(operand, prec)


def translate(xyz, amp, axis, ind=None, units='ang'):
    """Translates a set of atoms along a given vector.

    Parameters
    ----------
    xyz : (N, 3) array_like
        The atomic cartesian coordinates.
    amp : float
        The distance for translation.
    axis : array_like or str
        The axis of translation, parsed by :class:`VectorParser`.
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
    vp = VectorParser(xyz)
    u = vp(axis, unit=True)
    amp *= con.conv(units, 'ang')

    newxyz = np.copy(xyz)
    newxyz[ind] += amp * u
    return newxyz


def rotmat(ang, u, det=1, units='rad', xyz=None):
    r"""Returns the rotational matrix based on an angle and axis.

    A general rotational matrix in 3D can be formed given an angle and
    an axis by

    .. math::

        \mathbf{R} = \cos(a) \mathbf{I} + (\det(\mathbf{R}) -
        \cos(a)) \mathbf{u} \otimes \mathbf{u} + \sin(a) [\mathbf{u}]_\times

    for identity matrix **I**, angle *a*, axis **u**,
    outer product :math:`\otimes` and cross-product matrix
    :math:`[\mathbf{u}]_\times`.  Determinants of +1 and -1 give proper and
    improper rotation, respectively. Thus, :math:`\det(\mathbf{R}) = -1` and
    :math:`a = 0` is a reflection along the axis. Action of the rotational
    matrix occurs about the origin. See en.wikipedia.org/wiki/Rotation_matrix
    and http://scipp.ucsc.edu/~haber/ph251/rotreflect_17.pdf

    Parameters
    ----------
    ang : float
        The angle of rotation.
    u : array_like or str
        The axis of rotation, converted to a unit vector.
    det : int, optional
        The determinant of the matrix (1 or -1) used to specify proper
        and improper rotations. Default is 1.
    units : str, optional
        The units of angle for the rotation. Default is radians.
    xyz : (N, 3) array_like, optional
        The cartesian coordinates used in axis specification.

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

    u /= np.linalg.norm(u)
    amp = ang * con.conv(units, 'rad')
    ucross = np.array([[0, u[2], -u[1]], [-u[2], 0, u[0]], [u[1], -u[0], 0]])
    return (np.cos(amp) * np.eye(3) + np.sin(amp) * ucross +
            (det - np.cos(amp)) * np.outer(u, u))


def angax(rotmat, units='rad'):
    r"""Returns the angle, axis of rotation and determinant of a
    rotational matrix.

    Based on the form of **R**, it can be separated into symmetric
    and antisymmetric components with :math:`(r_{ij} + r_{ji})/2` and
    :math:`(r_{ij} - r_{ji})/2`, respectively. Then,

    .. math::

        r_{ii} = \cos(a) + u_i^2 (\det(\mathbf{R}) - \cos(a)),

        \cos(a) = (-\det(\mathbf{R}) + \sum_j r_{jj}) / 2 = 
        (\mathrm{tr}(\mathbf{R}) - \det(\mathbf{R})) / 2.

    From the expression for :math:`r_{ii}`, the magnitude of :math:`u_i`
    can be found

    .. math::

        |u_i| = \sqrt{\frac{1 + \det(\mathbf{R}) [2 r_{ii} -
        \mathrm{tr}(\mathbf{R})])}{3 - \det(\mathbf{R}) \mathrm{tr}(\mathbf{R})}},

    which satisfies :math:`u \cdot u = 1`. Note that if
    :math:`\det(\mathbf{R}) \mathrm{tr}(\mathbf{R}) = 3`, the axis is arbitrary
    (identity or inversion). Otherwise, the sign can be found from the
    antisymmetric component of **R**.

    .. math::

        u_i \sin(a) = (r_{jk} - r_{kj}) / 2, \quad i \neq j \neq k,

        \mathrm{sign}(u_i) = \mathrm{sign}(r_{jk} - r_{kj}),

    since :math:`\sin(a)` is positive in the range 0 to :math:`\pi`. :math:`i`,
    :math:`j` and :math:`k` obey the cyclic relation 3 -> 2 -> 1 -> 3 -> ...

    This fails when :math:`det(\mathbf{R}) \mathrm{tr}(\mathbf{R}) = -1`, in
    which case the symmetric component of **R** is used

    .. math::

        u_i u_j (\det(\mathbf{R}) - \cos(a)) = (r_{ij} + r_{ji}) / 2,

        \mathrm{sign}(u_i) \mathrm{sign}(u_j) = \det(\mathbf{R}) \mathrm{sign}(r_{ij} + r_{ji}).

    The signs can then be found by letting :math:`\mathrm{sign}(u_3) = +1`,
    since a rotation of :math:`pi` or a reflection are equivalent for
    antiparallel axes. See
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
    ang = con.arccos((tr - det) / 2) * con.conv('rad', units)
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
        The axis of rotation, parsed by :class:`VectorParser`.
    ind : array_like, optional
        List of atomic indices to specify which atoms are displaced. If
        ind is None (default) then all atoms are displaced.
    origin : (3,) array_like, optional
        The origin of rotation, parsed by :class:`VectorParser`. Default
        is the cartesian origin.
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

    vp = VectorParser(xyz)
    ax = vp(axis, unit=True)
    org = vp(origin)
    newxyz = xyz - org
    newxyz[ind] = newxyz[ind].dot(rotmat(ang, ax, det=det, units=units))
    return newxyz + org


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
    vp = VectorParser(xyz)
    test = vp(test_ax, unit=True)
    ref = vp(ref_ax, unit=True)
    if np.allclose(test, ref):
        return xyz
    elif np.allclose(test, -ref):
        rotax = np.array([0., 0., 1.])
        if np.allclose(test, rotax) or np.allclose(test, -rotax):
            rotax = np.array([0., 1., 0.])
        rotax -= np.dot(rotax, test) * test
        return rotate(xyz, np.pi, rotax, ind=ind, origin=origin)
    else:
        angle = con.arccos(np.dot(test, ref))
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
