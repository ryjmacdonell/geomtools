"""
Constant molecular properties to be used in other modules.

Arrays are ordered by atomic number for convenience. Atomic symbols are case
sensitive. This module should not depend on other geomtools modules.

Unit types
----------
Length: Angstrom (ang), Bohr (bohr), picometre (pm), nanometre (nm)
Angle: radian (rad), degree (deg)
Time: femtosecond (fs), picosecond (ps), atomic unit (au)
Mass: atomic mass unit (amu), electron mass (me), proton mass (mp), 
      kilogram (kg)
Energy: electron volt (ev), Hartree (har), kilocalorie per mole (kcm),
        kilojoule per mole (kjm), reciprocal centimetre (cm)
"""
import sys
import numpy as np


# Global constants
sym = np.array(['X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'])
mass = np.array([0.00000000, 1.00782504, 4.00260325, 7.01600450, 9.01218250,
                 11.00930530, 12.00000000, 14.00307401, 15.99491464,
                 18.99840325, 19.99243910, 22.98976970, 23.98504500,
                 26.98154130, 27.97692840, 30.97376340, 31.97207180,
                 34.96885273, 39.96238310])
covrad = np.array([0.000, 0.320, 1.600, 0.680, 0.352, 0.832, 0.720, 0.680,
                   0.680, 0.640, 1.120, 0.972, 1.100, 1.352, 1.200, 1.036,
                   1.020, 1.000, 1.568])
lenunits = {'ang':1., 'bohr':1./0.52917721, 'pm':100., 'nm':0.1}
angunits = {'rad':1., 'deg':180./np.pi}
timunits = {'fs':1., 'ps':0.001, 'au':0.024188843}
masunits = {'amu':1., 'me':5.48579909e-4, 'mp':1.00727647, 'kg':1.66053904e-27}
eneunits = {'ev':1., 'har':1./27.21138505, 'kcm':23.061, 'kjm':96.485, 
            'cm':8065.5}


def get_num(elem):
    """Returns atomic number from atomic symbol.

    Takes advantage of the fact that sym indices match atomic numbers. Input
    can be a single atom or a list of atoms.
    """
    if isinstance(elem, str):
        return np.where(sym == elem)[0][0]
    else:
        return np.array([np.where(sym == atm)[0][0] for atm in elem])


def get_mass(elem):
    """Returns atomic mass from atomic symbol."""
    return mass[get_num(elem)]


def get_covrad(elem):
    """Returns covalent radius from atomic symbol."""
    return covrad[get_num(elem)]


def conv(old_units, new_units):
    """Returns conversion factor from old units to new units."""
    if old_units == new_units:
        return 1.
    for unittype in [lenunits, angunits, timunits, masunits, eneunits]:
        if old_units in unittype and new_units in unittype:
            return unittype[new_units] / unittype[old_units]

    raise ValueError('Units \'{}\' and \'{}\' unrecognized or '
                     'not of same unit type'.format(old_units, new_units))
