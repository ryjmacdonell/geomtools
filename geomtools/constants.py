"""
Constant molecular properties to be used in other modules.

Arrays are ordered by atomic number for convenience. Atomic symbols are case
sensitive. This module should not depend on other geomtools modules.
"""
import sys
import numpy as np


# Global constants
a0 = 0.52917721092
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
lenunits = {'ang':1., 'au':1./a0, 'pm':100., 'nm':0.1}
angunits = {'rad':1., 'deg':180./np.pi}


def get_num(elem):
    """Returns atomic number from atomic symbol.

    Takes advantage of the fact that sym indices match atomic numbers. Input
    can be a single atom or a list of atoms.
    """
    global sym
    if isinstance(elem, str):
        return np.where(sym == elem)[0][0]
    else:
        return np.array([np.where(sym == atm)[0][0] for atm in elem])


def get_mass(elem):
    """Returns atomic mass from atomic symbol."""
    global mass
    return mass[get_num(elem)]


def get_covrad(elem):
    """Returns covalent radius from atomic symbol."""
    global covrad
    return covrad[get_num(elem)]


def unit_convert(old_units, new_units):
    """Returns conversion factor from old units to new units."""
    if old_units == new_units:
        return 1.
    if old_units in lenunits and new_units in lenunits:
        return lenunits[new_units] / lenunits[old_units]
    elif old_units in angunits and new_units in angunits:
        return angunits[new_units] / angunits[old_units]
    else:
        raise ValueError('Units \'{}\' and \'{}\' unrecognized or '
                         'not of same unit type'.format(old_units, new_units))
