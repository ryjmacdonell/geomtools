"""
Constant molecular properties to be used in other modules.

Arrays are ordered by atomic number for convenience. Atomic symbols are case
sensitive. This module should not depend on other modules.

Unit types
----------
Length: Angstrom (ang), Bohr (bohr), picometre (pm), nanometre (nm)

Angle: radian (rad), degree (deg)

Time: femtosecond (fs), picosecond (ps), atomic unit (au)

Mass: atomic mass unit (amu), electron mass (me), proton mass (mp),
kilogram (kg)

Energy: electron volt (ev), Hartree (har), kilocalorie per mole (kcm),
kilojoule per mole (kjm), reciprocal centimetre (cm)

For all types, 'auto' will give the default unit.

Attributes
----------
sym : ndarray
    List of atomic symbols up to Krypton. The ordering (with the
    exception of deuterium) yields the correct atomic number from
    ``sym.index(elem)``.
mass : ndarray
    List of atomic masses corresponding to the elements in `sym`.
covrad : ndarray
    List of covalent radii corresponding to the elements in `sym`.
lenunits : dict
    Dictionary of units of length and their conversions from the
    default (angstroms).
angunits : dict
    Dictionary of units of angle and their conversions from the
    default (radians).
timunits : dict
    Dictionary of units of time and their conversions from the
    default (femtoseconds).
masunits : dict
    Dictionary of units of mass and their conversions from the
    default (atomic mass units).
eneunits : dict
    Dictionary of units of energy and their conversion from the
    default (electron volts).
"""
import numpy as np


# Global constants
sym = np.array(['X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'])
mass = np.array([0.00000000, 1.00782504, 4.00260325, 7.01600450, 9.01218250,
                 11.00930530, 12.00000000, 14.00307401, 15.99491464,
                 18.99840325, 19.99243910, 22.98976970, 23.98504500,
                 26.98154130, 27.97692840, 30.97376340, 31.97207180,
                 34.96885273, 39.96238310, 38.96370790, 39.0983,
                 40.078, 44.955908, 47.867, 50.9415, 51.9961, 54.938044,
                 55.845, 58.933194, 58.6934, 63.546, 65.38, 69.723, 72.630,
                 74.921595, 78.971, 79.904, 83.798])
covrad = np.array([0.000, 0.320, 1.600, 0.680, 0.352, 0.832, 0.720, 0.680,
                   0.680, 0.640, 1.120, 0.972, 1.100, 1.352, 1.200, 1.036,
                   1.020, 1.000, 1.568, 1.328, 0.992, 1.440, 1.472, 1.328,
                   1.352, 1.352, 1.340, 1.328, 1.620, 1.520, 1.448, 1.220,
                   1.168, 1.208, 1.220, 1.208, 1.600])
lenunits = dict(auto=1., ang=1., bohr=1./0.52917721, pm=100., nm=0.1)
angunits = dict(auto=1., rad=1., deg=180./np.pi)
timunits = dict(auto=1., fs=1., ps=0.001, au=0.024188843)
masunits = dict(auto=1., amu=1., me=5.48579909e-4, mp=1.00727647,
                kg=1.66053904e-27)
eneunits = dict(auto=1., ev=1., har=1./27.21138505, kcm=23.061, kjm=96.485,
                cm=8065.5)


def get_num(elem):
    """Returns atomic number from atomic symbol.

    Takes advantage of the fact that sym indices match atomic numbers.

    Parameters
    ----------
    elem : str or array_like
        The atomic symbol(s) to be parsed.

    Returns
    -------
    int or ndarray
        The atomic numbers corresponding to each symbol.
    """
    if isinstance(elem, str):
        return _find_index(elem)
    else:
        for atm in elem:
            if atm not in sym and atm[0] not in ['X', 'D']:
                raise ValueError('Unrecognized atomic symbol \'' + atm +
                                 '\'. Use X prefix for dummy atoms.')
        return np.array([_find_index(atm) for atm in elem])


def get_mass(elem):
    """Returns atomic mass from atomic symbol.

    Parameters
    ----------
    elem : str of array_like
        The atomic symbol(s) to be parsed.

    Returns
    -------
    float or ndarray
        The atomic masses corresponding to each symbol.
    """
    return mass[get_num(elem)]


def get_covrad(elem):
    """Returns covalent radius from atomic symbol.

    Parameters
    ----------
    elem : str of array_like
        The atomic symbol(s) to be parsed.

    Returns
    -------
    float or ndarray
        The atomic covalent radii corresponding to each symbol.
    """
    return covrad[get_num(elem)]


def unit_vec(v):
    """Returns a unit vector aligned with a given vector.

    Parameters
    ---------
    v : array_like
        The input, un-normalized vector.

    Returns
    -------
    ndarray
        The normalized (unit) vector."""
    vlen = np.linalg.norm(v)
    if np.isclose(vlen, 0):
        raise ValueError('Cannot make unit vector from zero vector.')
    else:
        return v / vlen


def arccos(val):
    """Returns the arccosine of an angle allowing for numerical errors.

    NumPy's arccos function is defined for the range [-1, 1], but
    returns NaN for :math:`|x| = 1 + \delta`, where :math:`\delta` is
    small.  This can be avoided by checking for limiting cases with
    ``numpy.isclose``.

    Parameters
    ----------
    val : float
        The x-coordinate on the unit circle.

    Returns
    -------
    float
        The angle intersecting the unit circle at x = val.
    """
    if np.isclose(val, -1):
        return np.pi
    elif np.isclose(val, 1):
        return 0.
    else:
        return np.arccos(val)


def conv(old='auto', new='auto'):
    """Returns conversion factor from old units to new units.

    Parameters
    ----------
    old : str, optional
        The units to be converted from. See different units types for
        defaults.
    new : str, optional
        The units to be converted to. See different units types for
        defaults.

    Returns
    -------
    float
        The conversion factor, new_units / old_units.
    """
    if old == new:
        return 1.
    for unittype in [lenunits, angunits, timunits, masunits, eneunits]:
        if old in unittype and new in unittype:
            return unittype[new] / unittype[old]

    raise ValueError('Units \'{}\' and \'{}\' unrecognized or '
                     'not of same unit type'.format(old, new))


def _find_index(string):
    """Determines if dummy or regular atom and returns index.

    Parameters
    ----------
    string : str
        The atomic symbol.

    Returns
    -------
    int
        The atomic number of the given atomic symbol.
    """
    if string[0] == 'X':
        return 0
    elif string  == 'D':
        return 1
    else:
        return np.where(sym == string)[0][0]
