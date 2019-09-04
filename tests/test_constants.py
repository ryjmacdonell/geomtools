"""
Tests for the constants module
"""
import pytest
import numpy as np
import gimbal.constants as con


def test_symbol_lookup():
    assert con.sym[0] == 'X'


def test_mass_lookup():
    assert np.isclose(con.mass[0], 0.)


def test_covrad_lookup():
    assert np.isclose(con.covrad[0], 0.)


def test_lenunits_lookup():
    assert np.isclose(con.lenunits['auto'], 1.)


def test_angunits_lookup():
    assert np.isclose(con.angunits['auto'], 1.)


def test_timunits_lookup():
    assert np.isclose(con.timunits['auto'], 1.)


def test_masunits_lookup():
    assert np.isclose(con.masunits['auto'], 1.)


def test_eneunits_lookup():
    assert np.isclose(con.eneunits['auto'], 1.)


def test_get_num_single():
    assert con.get_num('H') == 1


def test_get_num_deuterium():
    assert con.get_num('D') == 1


def test_get_num_list():
    assert np.all(con.get_num(['H', 'He', 'Li']) == np.array([1, 2, 3]))


def test_get_num_fails():
    with pytest.raises(ValueError, match=r'Unrecognized atomic symbol .*'):
        con.get_num(['J', 'L', 'M'])


def test_get_mass():
    assert np.isclose(con.get_mass('H'), 1.00782504)


def test_get_covrad():
    assert np.isclose(con.get_covrad('H'), 0.320)


def test_unit_vec():
    vec_len = np.linalg.norm(con.unit_vec([1., -1., 2.]))
    assert np.isclose(vec_len, 1.)


def test_unit_vec_fails():
    with pytest.raises(ValueError, match=r'Cannot make unit vector from .*'):
        uvec = con.unit_vec(np.zeros(3))


def test_arccos_plusone():
    ang = con.arccos(1 + 1e-10)
    assert np.isclose(ang, 0)


def test_arccos_minusone():
    ang = con.arccos(-1 - 1e-10)
    assert np.isclose(ang, np.pi)


def test_conv_unit():
    assert np.isclose(con.conv('auto', 'auto'), 1.)


def test_conv_len():
    assert np.isclose(con.conv('ang', 'pm'), 100.)


def test_conv_ang():
    assert np.isclose(con.conv('deg', 'rad'), np.pi/180.)


def test_conv_tim():
    assert np.isclose(con.conv('ps', 'fs'), 1e3)


def test_conv_mas():
    assert np.isclose(con.conv('me', 'mp'), 1836.15267981)


def test_conv_ene():
    assert np.isclose(con.conv('har', 'ev'), 27.21138505)


def test_conv_fails():
    with pytest.raises(ValueError, match=r'.* not of same unit type'):
        con.conv('ev', 'fs')
