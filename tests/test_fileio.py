"""
Tests for the fileio module.
"""
import pytest
import numpy as np
import gimbal.fileio as fileio
from examples import Geometries as eg
from examples import Files as ef


def test_read_xyz_bohr(tmpdir):
    f = ef.tmpf(tmpdir, ef.xyz_bohr)
    elem, xyz, vec, com = fileio.read_xyz(f, units='bohr')
    assert np.all(elem == eg.ch4[0])
    assert np.allclose(xyz, eg.ch4[1])


def test_read_xyz_comment(tmpdir):
    f = ef.tmpf(tmpdir, ef.xyz_novec)
    elem, xyz, vec, com = fileio.read_xyz(f, hascom=True)
    assert np.all(elem == eg.ch4[0])
    assert np.allclose(xyz, eg.ch4[1])
    assert com == 'comment line'


def test_read_xyz_vector(tmpdir):
    f = ef.tmpf(tmpdir, ef.xyz_vec)
    elem, xyz, vec, com = fileio.read_xyz(f, hasvec=True)
    assert np.all(elem == eg.ch4[0])
    assert np.allclose(xyz, eg.ch4[1])
    assert np.allclose(vec, np.ones((5, 3)))


def test_read_xyz_vector_empty(tmpdir):
    f = ef.tmpf(tmpdir, ef.xyz_novec)
    elem, xyz, vec, com = fileio.read_xyz(f, hasvec=True)
    assert np.all(elem == eg.ch4[0])
    assert np.allclose(xyz, eg.ch4[1])
    assert np.allclose(vec, np.zeros((5, 3)))


def test_read_xyz_wrong_format(tmpdir):
    f = ef.tmpf(tmpdir, ef.col_nocom)
    with pytest.raises(IOError, match=r'geometry not in XYZ format.'):
        elem, xyz, vec, com = fileio.read_xyz(f)


def test_read_col_comment(tmpdir):
    f = ef.tmpf(tmpdir, ef.col_com)
    elem, xyz, vec, com = fileio.read_col(f, hascom=True)
    assert np.all(elem == eg.ch4[0])
    assert np.allclose(xyz, eg.ch4[1])
    assert com == 'comment line'


def test_read_col_hasvec(tmpdir):
    f = ef.tmpf(tmpdir, ef.col_nocom)
    elem, xyz, vec, com = fileio.read_col(f, hasvec=True)
    assert np.all(elem == eg.ch4[0])
    assert np.allclose(xyz, eg.ch4[1])
    assert np.allclose(vec, np.zeros_like(xyz))


def test_read_col_wrong_format(tmpdir):
    f = ef.tmpf(tmpdir, ef.gdat_bohr)
    with pytest.raises(IOError, match=r'geometry not in COLUMBUS format.'):
        elem, xyz, vec, com = fileio.read_col(f)


def test_read_gdat_comment(tmpdir):
    f = ef.tmpf(tmpdir, ef.gdat_bohr)
    elem, xyz, vec, com = fileio.read_gdat(f, hascom=True)
    assert np.all(elem == eg.ch4[0])
    assert np.allclose(xyz, eg.ch4[1])
    assert com == 'comment line'


def test_read_gdat_vector(tmpdir):
    f = ef.tmpf(tmpdir, ef.gdat_bohr)
    elem, xyz, vec, com = fileio.read_gdat(f, hasvec=True)
    assert np.all(elem == eg.ch4[0])
    assert np.allclose(xyz, eg.ch4[1])
    assert np.allclose(vec, np.ones((5, 3)))


def test_read_gdat_wrong_format(tmpdir):
    f = ef.tmpf(tmpdir, ef.zmt_nocom)
    with pytest.raises(IOError, match=r'geometry not in Geometry.dat .*'):
        elem, xyz, vec, com = fileio.read_gdat(f)


def test_read_zmt_comment(tmpdir):
    f = ef.tmpf(tmpdir, ef.zmt_com)
    elem, xyz, vec, com = fileio.read_zmt(f, hascom=True)
    assert np.all(elem == eg.ch4[0])
    assert np.allclose(xyz, eg.ch4_zmt[1], atol=1e-6)
    assert com == 'comment line'


def test_read_zmt_hasvec(tmpdir):
    f = ef.tmpf(tmpdir, ef.zmt_nocom)
    elem, xyz, vec, com = fileio.read_zmt(f, hasvec=True)
    assert np.all(elem == eg.ch4[0])
    assert np.allclose(xyz, eg.ch4_zmt[1], atol=1e-6)
    assert np.allclose(vec, np.zeros_like(xyz))


def test_read_zmt_var(tmpdir):
    f = ef.tmpf(tmpdir, ef.zmtvar_nocom)
    elem, xyz, vec, com = fileio.read_zmt(f)
    assert np.all(elem == eg.ch4_zmt[0])
    assert np.allclose(xyz, eg.ch4_zmt[1], atol=1e-6)


def test_read_zmt_wrong_format(tmpdir):
    f = ef.tmpf(tmpdir, ef.traj_nocom)
    with pytest.raises(IOError, match=r'geometry not in Z-matrix .*'):
        elem, xyz, vec, com = fileio.read_zmt(f)


def test_read_traj_no_elem(tmpdir):
    f = ef.tmpf(tmpdir, ef.traj_nocom)
    elem, xyz, vec, com = fileio.read_traj(f)
    assert np.all(elem == 5*['X'])
    assert np.allclose(xyz, eg.ch4[1], atol=1e-4)


def test_read_traj_comment(tmpdir):
    f = ef.tmpf(tmpdir, ef.traj_com)
    elem, xyz, vec, com = fileio.read_traj(f, hascom=True, elem=eg.ch4[0])
    assert np.all(elem == eg.ch4[0])
    assert np.allclose(xyz, eg.ch4[1], atol=1e-4)
    assert com == 'comment line'


def test_read_traj_vector(tmpdir):
    f = ef.tmpf(tmpdir, ef.traj_nocom)
    elem, xyz, vec, com = fileio.read_traj(f, hasvec=True, elem=eg.ch4[0])
    assert np.all(elem == eg.ch4[0])
    assert np.allclose(xyz, eg.ch4[1], atol=1e-4)
    assert np.allclose(vec, np.ones((5, 3)), atol=1e-4)


def test_read_traj_time(tmpdir):
    f = ef.tmpf(tmpdir, ef.traj_time)
    elem, xyz, vec, com = fileio.read_traj(f, hasvec=True, time=1.)
    assert np.all(elem == 5*['X'])
    assert np.allclose(xyz, eg.ch4[1], atol=1e-4)
    assert np.allclose(vec, np.ones((5, 3)), atol=1e-4)


def test_read_traj_autocom(tmpdir):
    f = ef.tmpf(tmpdir, ef.traj_nocom)
    elem, xyz, vec, com = fileio.read_traj(f, autocom=True)
    assert np.all(elem == 5*['X'])
    assert np.allclose(xyz, eg.ch4[1], atol=1e-4)
    assert com == 't=    0.00, state=   2, a^2=    0.2500'


def test_read_traj_empty_line(tmpdir):
    f = ef.tmpf(tmpdir, '\n')
    with pytest.raises(IOError, match=r'empty line provided'):
        elem, xyz, vec, com = fileio.read_traj(f)


def test_read_traj_wrong_format(tmpdir):
    f = ef.tmpf(tmpdir, ef.xyz_novec)
    with pytest.raises(IOError, match=r'geometry not in trajectory .*'):
        elem, xyz, vec, com = fileio.read_traj(f)


def test_read_auto_end_of_file(tmpdir):
    f = ef.tmpf(tmpdir, '')
    with pytest.raises(IOError, match=r'end of file'):
        elem, xyz, vec, com = fileio.read_auto(f)


def test_read_auto_only_comment(tmpdir):
    f = ef.tmpf(tmpdir, 'comment line\n')
    with pytest.raises(IOError, match=r'cannot have comment with single .*'):
        elem, xyz, vec, com = fileio.read_auto(f, hascom=True)


def test_read_auto_zmt_one_line(tmpdir):
    f = ef.tmpf(tmpdir, 'He\n')
    elem, xyz, vec, com = fileio.read_auto(f)
    assert np.all(elem == ['He'])
    assert np.allclose(xyz, np.zeros((1, 3)))


def test_read_auto_col_one_line(tmpdir):
    he_col = (' He    2.0    0.00000000    0.00000000    0.00000000' +
              '    4.00260325\n')
    f = ef.tmpf(tmpdir, he_col)
    elem, xyz, vec, com = fileio.read_auto(f)
    assert np.all(elem == ['He'])
    assert np.allclose(xyz, np.zeros((1, 3)))


def test_read_auto_traj_one_line(tmpdir):
    f = ef.tmpf(tmpdir, ef.traj_nocom)
    elem, xyz, vec, com = fileio.read_auto(f)
    assert np.all(elem == 5*['X'])
    assert np.allclose(xyz, eg.ch4[1], atol=1e-4)


def test_read_auto_unrecognized_one_line(tmpdir):
    f = ef.tmpf(tmpdir, '123 abc 456 def\n')
    with pytest.raises(IOError, match=r'single line input in unrecognized .*'):
        elem, xyz, vec, com = fileio.read_auto(f)


def test_read_auto_zmt_one_line_comment(tmpdir):
    f = ef.tmpf(tmpdir, 'comment line\nHe\n')
    elem, xyz, vec, com = fileio.read_auto(f, hascom=True)
    assert np.all(elem == ['He'])
    assert np.allclose(xyz, np.zeros((1, 3)))
    assert com == 'comment line'


def test_read_auto_col_one_line_comment(tmpdir):
    he_col = ('comment line\n' +
              ' He    2.0    0.00000000    0.00000000    0.00000000' +
              '    4.00260325\n')
    f = ef.tmpf(tmpdir, he_col)
    elem, xyz, vec, com = fileio.read_auto(f, hascom=True)
    assert np.all(elem == ['He'])
    assert np.allclose(xyz, np.zeros((1, 3)))
    assert com == 'comment line'


def test_read_auto_traj_one_line_comment(tmpdir):
    f = ef.tmpf(tmpdir, ef.traj_com)
    elem, xyz, vec, com = fileio.read_auto(f, hascom=True)
    assert np.all(elem == 5*['X'])
    assert np.allclose(xyz, eg.ch4[1], atol=1e-4)
    assert com == 'comment line'


def test_read_auto_unrecognized_one_line_comment(tmpdir):
    f = ef.tmpf(tmpdir, 'comment line\n123 abc 456 def\n')
    with pytest.raises(IOError, match=r'single line input in unrecognized .*'):
        elem, xyz, vec, com = fileio.read_auto(f, hascom=True)


def test_read_auto_xyz(tmpdir):
    f = ef.tmpf(tmpdir, ef.xyz_novec)
    elem, xyz, vec, com = fileio.read_auto(f)
    assert np.all(elem == eg.ch4[0])
    assert np.allclose(xyz, eg.ch4[1])


def test_read_auto_col(tmpdir):
    f = ef.tmpf(tmpdir, ef.col_nocom)
    elem, xyz, vec, com = fileio.read_auto(f)
    assert np.all(elem == eg.ch4[0])
    assert np.allclose(xyz, eg.ch4[1])


def test_read_auto_col_comment(tmpdir):
    f = ef.tmpf(tmpdir, ef.col_com)
    elem, xyz, vec, com = fileio.read_auto(f, hascom=True)
    assert np.all(elem == eg.ch4[0])
    assert np.allclose(xyz, eg.ch4[1])
    assert com == 'comment line'


def test_read_auto_gdat(tmpdir):
    f = ef.tmpf(tmpdir, ef.gdat_bohr)
    elem, xyz, vec, com = fileio.read_auto(f)
    assert np.all(elem == eg.ch4[0])
    assert np.allclose(xyz, eg.ch4[1])


def test_read_auto_zmt(tmpdir):
    f = ef.tmpf(tmpdir, ef.zmt_nocom)
    elem, xyz, vec, com = fileio.read_auto(f)
    assert np.all(elem == eg.ch4[0])
    assert np.allclose(xyz, eg.ch4_zmt[1], atol=1e-6)


def test_read_auto_zmt_comment(tmpdir):
    f = ef.tmpf(tmpdir, ef.zmt_com)
    elem, xyz, vec, com = fileio.read_auto(f, hascom=True)
    assert np.all(elem == eg.ch4[0])
    assert np.allclose(xyz, eg.ch4_zmt[1], atol=1e-6)
    assert com == 'comment line'


def test_read_auto_traj(tmpdir):
    f = ef.tmpf(tmpdir, ef.traj_time)
    elem, xyz, vec, com = fileio.read_auto(f)
    assert np.all(elem == 5*['X'])
    assert np.allclose(xyz, eg.ch4[1], atol=1e-4)


def test_read_auto_unrecognized_multi_lines(tmpdir):
    f = ef.tmpf(tmpdir, 'abc 123 def 456\nabc def 123 456\n123 456 abc def\n')
    with pytest.raises(IOError, match=r'unrecognized file format'):
        elem, xyz, vec, com = fileio.read_auto(f)


def test_read_multiple_no_geoms(tmpdir):
    f = ef.tmpf(tmpdir, '\n')
    with pytest.raises(IOError, match=r'no geometries read from input files'):
        moldat = fileio.read_multiple(f)


def test_write_xyz_vec(tmpdir):
    f = tmpdir.join('tmp.geom')
    fileio.write_xyz(f.open(mode='w'), eg.ch4[0], eg.ch4[1],
                     vec=np.ones((5, 3)))
    assert f.read() == ef.xyz_vec.replace('comment line', '')


def test_write_xyz_bohr(tmpdir):
    f = tmpdir.join('tmp.geom')
    fileio.write_xyz(f.open(mode='w'), eg.ch4[0], eg.ch4[1], units='bohr')
    assert f.read() == ef.xyz_bohr.replace('comment line', '')


def test_write_col_comment(tmpdir):
    f = tmpdir.join('tmp.geom')
    fileio.write_col(f.open(mode='w'), eg.ch4[0], eg.ch4[1],
                     comment='comment line')
    assert f.read() == ef.col_com


def test_write_gdat_vec(tmpdir):
    f = tmpdir.join('tmp.geom')
    fileio.write_gdat(f.open(mode='w'), eg.ch4[0], eg.ch4[1],
                      vec=np.ones((5, 3)))
    assert f.read() == ef.gdat_bohr.replace('comment line', '')


def test_write_zmt_comment(tmpdir):
    f = tmpdir.join('tmp.geom')
    fileio.write_zmt(f.open(mode='w'), eg.ch4[0], eg.ch4[1],
                     comment='comment line')
    assert f.read() == ef.zmt_com


def test_write_zmtvar_comment(tmpdir):
    f = tmpdir.join('tmp.geom')
    fileio.write_zmtvar(f.open(mode='w'), eg.ch4[0], eg.ch4[1],
                        comment='comment line')
    assert f.read() == ef.zmtvar_com


def test_write_traj_vec(tmpdir):
    f = tmpdir.join('tmp.geom')
    fileio.write_traj(f.open(mode='w'), eg.ch4[0], eg.ch4[1],
                      vec=np.ones((5, 3)))
    soln = ef.traj_nocom.replace('0.4000    0.3000    0.2500    2.0000',
                                 '0.0000    0.0000    0.0000    0.0000')
    assert f.read() == soln


def test_write_traj_comment(tmpdir):
    f = tmpdir.join('tmp.geom')
    fileio.write_traj(f.open(mode='w'), eg.ch4[0], eg.ch4[1],
                      comment='comment line')
    soln = ef.traj_com.replace('0.4000    0.3000    0.2500    2.0000',
                               '0.0000    0.0000    0.0000    0.0000')
    assert f.read() == soln.replace(' 1.0000', ' 0.0000')


def test_write_auto_col(tmpdir):
    f = tmpdir.join('tmp.col')
    fileio.write_auto(f.open(mode='w'), eg.ch4[0], eg.ch4[1])
    assert f.read() == ef.col_nocom


def test_write_auto_gdat(tmpdir):
    f = tmpdir.join('tmp.dat')
    fileio.write_auto(f.open(mode='w'), eg.ch4[0], eg.ch4[1])
    soln = ef.gdat_bohr.replace('comment line', '').replace(' 1.000', ' 0.000')
    assert f.read() == soln


def test_write_auto_zmt(tmpdir):
    f = tmpdir.join('tmp.zmt')
    fileio.write_auto(f.open(mode='w'), eg.ch4[0], eg.ch4[1])
    assert f.read() == ef.zmt_nocom


def test_write_auto_zmtvar(tmpdir):
    f = tmpdir.join('tmp.zmtvar')
    fileio.write_auto(f.open(mode='w'), eg.ch4[0], eg.ch4[1])
    assert f.read() == ef.zmtvar_nocom


def test_write_auto_traj(tmpdir):
    f = tmpdir.join('tmp.tj')
    fileio.write_traj(f.open(mode='w'), eg.ch4[0], eg.ch4[1])
    soln = ef.traj_nocom.replace('0.4000    0.3000    0.2500    2.0000',
                                 '0.0000    0.0000    0.0000    0.0000')
    assert f.read() == soln.replace(' 1.0000', ' 0.0000')


def test_write_auto_xyz(tmpdir):
    f = tmpdir.join('tmp.xyz')
    fileio.write_auto(f.open(mode='w'), eg.ch4[0], eg.ch4[1])
    assert f.read() == ef.xyz_novec.replace('comment line', '')


def test_convert_xyz_to_col(tmpdir):
    f1 = tmpdir.join('tmp.xyz')
    f1.write(ef.xyz_novec)
    f2 = tmpdir.join('tmp.col')
    fileio.convert(str(f1.realpath()), str(f2.realpath()),
                   infmt='xyz', outfmt='col')
    assert f2.read() == ef.col_nocom


def test_convert_xyz_to_zmt_inunits_bohr(tmpdir):
    f1 = tmpdir.join('tmp.xyz')
    f1.write(ef.xyz_bohr.replace('88861', '8886072'))
    f2 = tmpdir.join('tmp.col')
    fileio.convert(str(f1.realpath()), str(f2.realpath()), inunits='bohr')
    assert f2.read() == ef.col_nocom


def test_convert_xyz_to_gdat_outunits_angstrom(tmpdir):
    f1 = tmpdir.join('tmp.xyz')
    f1.write(ef.xyz_vec)
    f2 = tmpdir.join('tmp.dat')
    fileio.convert(str(f1.realpath()), str(f2.realpath()), outunits='ang',
                   hasvec=True)
    assert f2.read() == ef.gdat_ang


def test_convert_xyz_to_zmtvar_hascom(tmpdir):
    f1 = tmpdir.join('tmp.xyz')
    f1.write(ef.xyz_novec)
    f2 = tmpdir.join('tmp.zmtvar')
    fileio.convert(str(f1.realpath()), str(f2.realpath()), hascom=True)
    assert f2.read() == ef.zmtvar_com


def test_convert_xyz_to_traj_hasvec(tmpdir):
    f1 = tmpdir.join('tmp.xyz')
    f1.write(ef.xyz_vec)
    f2 = tmpdir.join('tmp.tj')
    fileio.convert(str(f1.realpath()), str(f2.realpath()), hasvec=True)
    soln = ef.traj_nocom.replace('0.4000    0.3000    0.2500    2.0000',
                                 '0.0000    0.0000    0.0000    0.0000')
    assert f2.read() == soln


def test_get_optargs_no_default():
    args = ['-u', 'arg1', 'arg2', '-v']
    val1 = fileio.get_optarg(args, '-u')
    val2 = fileio.get_optarg(args, '-v')
    assert val1 is True
    assert val2 is True
    assert args == ['arg1', 'arg2']


def test_get_optargs_set_default():
    args = ['-u', 'optarg1', 'arg1', 'arg2', '-v', 'optarg2']
    val1 = fileio.get_optarg(args, '-u', default=None)
    val2 = fileio.get_optarg(args, '-v', default=None)
    assert val1 == 'optarg1'
    assert val2 == 'optarg2'
    assert args == ['arg1', 'arg2']


def test_get_optargs_not_found():
    args = ['arg1', 'arg2']
    val1 = fileio.get_optarg(args, '-u')
    val2 = fileio.get_optarg(args, '-v', default=None)
    assert val1 is False
    assert val2 is None
    assert args == ['arg1', 'arg2']


def test_valvar_fails(tmpdir):
    tdict = dict(a = 1.)
    with pytest.raises(KeyError, match=r'.* not found in variable list'):
        fileio._valvar('b', tdict)
