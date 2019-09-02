"""
Tests for the measure module.
"""
import pytest
import numpy as np
import gimbal.measure as measure
from examples import Geometries as eg


def test_stre_default():
    stre = measure.stre(eg.c2h4_ms[1], 0, 1)
    assert np.isclose(stre, 1.33496)


def test_stre_bohr():
    stre = measure.stre(eg.c2h4_ms[1], 0, 1, units='bohr')
    assert np.isclose(stre, 2.52270879)


def test_bend_default():
    bend = measure.bend(eg.c2h4_ms[1], 4, 1, 5)
    assert np.isclose(bend, 2.03464239)


def test_bend_degrees():
    bend = measure.bend(eg.c2h4_ms[1], 4, 1, 5, units='deg')
    assert np.isclose(bend, 116.57642205)


def test_tors_default():
    tors = measure.tors(eg.c2h4_ms[1], 2, 0, 1, 5)
    assert np.isclose(tors, -3.01462393)


def test_tors_degrees():
    tors = measure.tors(eg.c2h4_ms[1], 2, 0, 1, 5, units='deg')
    assert np.isclose(tors, -172.72522822)


def test_tors_absv():
    tors = measure.tors(eg.c2h4_ms[1], 2, 0, 1, 5, absv=True)
    assert np.isclose(tors, 3.01462393)


def test_oop_default():
    oop = measure.oop(eg.c2h4_ms[1], 1, 3, 2, 0)
    assert np.isclose(oop, -np.pi/4)


def test_oop_greater_than_pi():
    xyz = np.array([[    0.,             0.,             0.],
                    [-1./2.,             0., -np.sqrt(3)/2.],
                    [-1./2.,  np.sqrt(3)/2.,             0.],
                    [-1./2., -np.sqrt(3)/2.,             0.]])
    oop = measure.oop(xyz, 1, 2, 3, 0)
    assert np.isclose(oop, -2*np.pi/3)


def test_oop_degrees():
    oop = measure.oop(eg.c2h4_ms[1], 1, 3, 2, 0, units='deg')
    assert np.isclose(oop, -45.)


def test_oop_absv():
    oop = measure.oop(eg.c2h4_ms[1], 1, 3, 2, 0, absv=True)
    assert np.isclose(oop, np.pi/4)


def test_planeang_default():
    planeang = measure.planeang(eg.c2h4_ms[1], 2, 3, 0, 1, 4, 5)
    assert np.isclose(planeang, -1.93216345)


def test_planeang_degrees():
    planeang = measure.planeang(eg.c2h4_ms[1], 2, 3, 0, 1, 4, 5, units='deg')
    assert np.isclose(planeang, -110.70481105)


def test_planeang_absv():
    planeang = measure.planeang(eg.c2h4_ms[1], 2, 3, 0, 1, 4, 5, absv=True)
    assert np.isclose(planeang, 1.93216345)


def test_planetors_default():
    planetors = measure.planetors(eg.c2h4_ms[1], 2, 3, 0, 1, 4, 5)
    assert np.isclose(planetors, -2*np.pi/3)


def test_planetors_degrees():
    planetors = measure.planetors(eg.c2h4_ms[1], 2, 3, 0, 1, 4, 5, units='deg')
    assert np.isclose(planetors, -120.)


def test_planetors_absv():
    planetors = measure.planetors(eg.c2h4_ms[1], 2, 3, 0, 1, 4, 5, absv=True)
    assert np.isclose(planetors, 2*np.pi/3)


def test_edgetors_default():
    edgetors = measure.edgetors(eg.c2h4_ms[1], 2, 3, 0, 1, 4, 5)
    assert np.isclose(edgetors, -2.48199179)


def test_edgetors_degrees():
    edgetors = measure.edgetors(eg.c2h4_ms[1], 2, 3, 0, 1, 4, 5, units='deg')
    assert np.isclose(edgetors, -142.20765430)


def test_edgetors_absv():
    edgetors = measure.edgetors(eg.c2h4_ms[1], 2, 3, 0, 1, 4, 5, absv=True)
    assert np.isclose(edgetors, 2.48199179)
