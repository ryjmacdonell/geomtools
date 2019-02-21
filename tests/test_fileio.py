"""
Tests for the fileio module.
"""
import pytest
import numpy as np
import gimbal.fileio as fileio


ch4 = (['C', 'H', 'H', 'H', 'H'],
       np.array([[ 0.000000,  0.000000,  0.000000],
                 [ 0.629118,  0.629118,  0.629118],
                 [-0.629118, -0.629118,  0.629118],
                 [ 0.629118, -0.629118, -0.629118],
                 [-0.629118,  0.629118, -0.629118]]),
       np.ones((5, 3)))
zmt2xyz = np.array([[ 0.000000,  0.000000,  0.000000],
                    [ 0.000000,  0.000000,  1.089664],
                    [-1.027345,  0.000000, -0.363221],
                    [ 0.513673, -0.889707, -0.363221],
                    [ 0.513673,  0.889707, -0.363221]])
xyz_novec = """\
 5
comment line
C       0.000000    0.000000    0.000000
H       0.629118    0.629118    0.629118
H      -0.629118   -0.629118    0.629118
H       0.629118   -0.629118   -0.629118
H      -0.629118    0.629118   -0.629118
"""
xyz_vec = """\
 5
comment line
C       0.000000    0.000000    0.000000    1.000000    1.000000    1.000000
H       0.629118    0.629118    0.629118    1.000000    1.000000    1.000000
H      -0.629118   -0.629118    0.629118    1.000000    1.000000    1.000000
H       0.629118   -0.629118   -0.629118    1.000000    1.000000    1.000000
H      -0.629118    0.629118   -0.629118    1.000000    1.000000    1.000000
"""
col_nocom = """\
 C     6.0    0.00000000    0.00000000    0.00000000   12.00000000
 H     1.0    1.18886072    1.18886072    1.18886072    1.00782504
 H     1.0   -1.18886072   -1.18886072    1.18886072    1.00782504
 H     1.0    1.18886072   -1.18886072   -1.18886072    1.00782504
 H     1.0   -1.18886072    1.18886072   -1.18886072    1.00782504
"""
col_com = 'comment line\n' + col_nocom
gdat = """\
comment line
5
C     0.00000000E+00    0.00000000E+00    0.00000000E+00
H     1.18886072E+00    1.18886072E+00    1.18886072E+00
H    -1.18886072E+00   -1.18886072E+00    1.18886072E+00
H     1.18886072E+00   -1.18886072E+00   -1.18886072E+00
H    -1.18886072E+00    1.18886072E+00   -1.18886072E+00
     1.00000000E+00    1.00000000E+00    1.00000000E+00
     1.00000000E+00    1.00000000E+00    1.00000000E+00
     1.00000000E+00    1.00000000E+00    1.00000000E+00
     1.00000000E+00    1.00000000E+00    1.00000000E+00
     1.00000000E+00    1.00000000E+00    1.00000000E+00
"""
zmt_nocom = """\
C 
H   1    1.089664
H   2    1.779414  1   35.264390
H   3    1.779414  2   60.000000  1   35.264390
H   4    1.779414  3   60.000000  2  -70.528779
"""
zmt_com = 'comment line\n' + zmt_nocom
zmtvar_nocom = """\
C 
H   1  R1 
H   2  R2    1  A1 
H   3  R3    2  A2    1  T1 
H   4  R4    3  A3    2  T2 

R1   =     1.08966434
R2   =     1.77941442
A1   =    35.26438968
R3   =     1.77941442
A2   =    60.00000000
T1   =    35.26438968
R4   =     1.77941442
A3   =    60.00000000
T2   =   -70.52877937
"""
zmtvar_com = 'comment line\n' + zmtvar_nocom
traj_nocom = ('      0.00    0.0000    0.0000    0.0000    1.1889    1.1889' +
'    1.1889   -1.1889   -1.1889    1.1889    1.1889   -1.1889   -1.1889' +
'   -1.1889    1.1889   -1.1889    1.0000    1.0000    1.0000    1.0000' +
'    1.0000    1.0000    1.0000    1.0000    1.0000    1.0000    1.0000' +
'    1.0000    1.0000    1.0000    1.0000    0.0000    0.0000    0.0000' +
'    0.0000    0.0000\n')
traj_time = traj_nocom + traj_nocom.replace('0.00 ', '1.00 ')
traj_com = 'comment line\n' + traj_nocom


def tmpf(tmpdir, fname, contents):
    """Writes a temporary file to test reading."""
    fout = tmpdir.join(fname)
    fout.write(contents)
    return fout.open()


def test_read_xyz_bohr(tmpdir):
    f = tmpf(tmpdir, 'ch4.xyz', xyz_novec)
    elem, xyz, vec, com = fileio.read_xyz(f)
    assert np.all(elem == ch4[0])
    assert np.allclose(xyz, ch4[1])


def test_read_xyz_comment(tmpdir):
    f = tmpf(tmpdir, 'ch4.xyz', xyz_novec)
    elem, xyz, vec, com = fileio.read_xyz(f, hascom=True)
    assert np.all(elem == ch4[0])
    assert np.allclose(xyz, ch4[1])
    assert com == 'comment line'


def test_read_xyz_vector(tmpdir):
    f = tmpf(tmpdir, 'ch4.xyz', xyz_vec)
    elem, xyz, vec, com = fileio.read_xyz(f, hasvec=True)
    assert np.all(elem == ch4[0])
    assert np.allclose(xyz, ch4[1])
    assert np.allclose(vec, ch4[2])


def test_read_xyz_wrong_format(tmpdir):
    f = tmpf(tmpdir, 'ch4.col', col_nocom)
    with pytest.raises(IOError, match=r'geometry not in XYZ format.'):
        elem, xyz, vec, com = fileio.read_xyz(f)


def test_read_col_comment(tmpdir):
    f = tmpf(tmpdir, 'ch4.col', col_com)
    elem, xyz, vec, com = fileio.read_col(f, hascom=True)
    assert np.all(elem == ch4[0])
    assert np.allclose(xyz, ch4[1])
    assert com == 'comment line'


def test_read_col_wrong_format(tmpdir):
    f = tmpf(tmpdir, 'ch4.dat', gdat)
    with pytest.raises(IOError, match=r'geometry not in COLUMBUS format.'):
        elem, xyz, vec, com = fileio.read_col(f)


def test_read_gdat_comment(tmpdir):
    f = tmpf(tmpdir, 'ch4.dat', gdat)
    elem, xyz, vec, com = fileio.read_gdat(f, hascom=True)
    assert np.all(elem == ch4[0])
    assert np.allclose(xyz, ch4[1])
    assert com == 'comment line'


def test_read_gdat_vector(tmpdir):
    f = tmpf(tmpdir, 'ch4.dat', gdat)
    elem, xyz, vec, com = fileio.read_gdat(f, hasvec=True)
    assert np.all(elem == ch4[0])
    assert np.allclose(xyz, ch4[1])
    assert np.allclose(vec, ch4[2])


def test_read_gdat_wrong_format(tmpdir):
    f = tmpf(tmpdir, 'ch4.zmt', zmt_nocom)
    with pytest.raises(IOError, match=r'geometry not in Geometry.dat .*'):
        elem, xyz, vec, com = fileio.read_gdat(f)


def test_read_zmt_comment(tmpdir):
    f = tmpf(tmpdir, 'ch4.zmt', zmt_com)
    elem, xyz, vec, com = fileio.read_zmt(f, hascom=True)
    assert np.all(elem == ch4[0])
    assert np.allclose(xyz, zmt2xyz, atol=1e-6)
    assert com == 'comment line'


def test_read_zmt_wrong_format(tmpdir):
    f = tmpf(tmpdir, 'ch4.tj', traj_nocom)
    with pytest.raises(IOError, match=r'geometry not in Z-matrix .*'):
        elem, xyz, vec, com = fileio.read_zmt(f)


def test_read_traj_no_elem(tmpdir):
    f = tmpf(tmpdir, 'ch4.tj', traj_nocom)
    elem, xyz, vec, com = fileio.read_traj(f)
    assert np.all(elem == 5*['X'])
    assert np.allclose(xyz, ch4[1], atol=1e-4)


def test_read_traj_comment(tmpdir):
    f = tmpf(tmpdir, 'ch4.tj', traj_com)
    elem, xyz, vec, com = fileio.read_traj(f, hascom=True, elem=ch4[0])
    assert np.all(elem == ch4[0])
    assert np.allclose(xyz, ch4[1], atol=1e-4)
    assert com == 'comment line'


def test_read_traj_vector(tmpdir):
    f = tmpf(tmpdir, 'ch4.tj', traj_nocom)
    elem, xyz, vec, com = fileio.read_traj(f, hasvec=True, elem=ch4[0])
    assert np.all(elem == ch4[0])
    assert np.allclose(xyz, ch4[1], atol=1e-4)
    assert np.allclose(vec, ch4[2], atol=1e-4)


def test_read_traj_time(tmpdir):
    f = tmpf(tmpdir, 'ch4.tj', traj_time)
    elem, xyz, vec, com = fileio.read_traj(f, hasvec=True, time=1.)
    assert np.all(elem == 5*['X'])
    assert np.allclose(xyz, ch4[1], atol=1e-4)


def test_read_traj_wrong_format(tmpdir):
    f = tmpf(tmpdir, 'ch4.xyz', xyz_novec)
    with pytest.raises(IOError, match=r'geometry not in trajectory .*'):
        elem, xyz, vec, com = fileio.read_traj(f)


def test_read_auto_end_of_file(tmpdir):
    f = tmpf(tmpdir, 'nothing.xyz', '')
    with pytest.raises(IOError, match=r'end of file'):
        elem, xyz, vec, com = fileio.read_auto(f)


def test_read_auto_only_comment(tmpdir):
    f = tmpf(tmpdir, 'comment.xyz', 'comment line\n')
    with pytest.raises(IOError, match=r'cannot have comment with single .*'):
        elem, xyz, vec, com = fileio.read_auto(f, hascom=True)


def test_read_auto_zmt_one_line(tmpdir):
    f = tmpf(tmpdir, 'he.zmt', 'He\n')
    elem, xyz, vec, com = fileio.read_zmt(f)
    assert np.all(elem == ['He'])
    assert np.allclose(xyz, np.zeros((1, 3)))


def test_read_auto_col_one_line(tmpdir):
    he_col = (' He    2.0    0.00000000    0.00000000    0.00000000' +
              '    4.00260325\n')
    f = tmpf(tmpdir, 'he.col', he_col)
    elem, xyz, vec, com = fileio.read_auto(f)
    assert np.all(elem == ['He'])
    assert np.allclose(xyz, np.zeros((1, 3)))


def test_read_auto_traj_one_line(tmpdir):
    f = tmpf(tmpdir, 'ch4.tj', traj_nocom)
    elem, xyz, vec, com = fileio.read_auto(f)
    assert np.all(elem == 5*['X'])
    assert np.allclose(xyz, ch4[1], atol=1e-4)


def test_read_auto_unrecognized_one_line(tmpdir):
    f = tmpf(tmpdir, 'unrec.xyz', '123 abc 456 def\n')
    with pytest.raises(IOError, match=r'single line input in unrecognized .*'):
        elem, xyz, vec, com = fileio.read_auto(f)


def test_read_auto_zmt_one_line_comment(tmpdir):
    f = tmpf(tmpdir, 'he.zmt', 'comment line\nHe\n')
    elem, xyz, vec, com = fileio.read_auto(f, hascom=True)
    assert np.all(elem == ['He'])
    assert np.allclose(xyz, np.zeros((1, 3)))
    assert com == 'comment line'


def test_read_auto_col_one_line_comment(tmpdir):
    he_col = ('comment line\n' +
              ' He    2.0    0.00000000    0.00000000    0.00000000' +
              '    4.00260325\n')
    f = tmpf(tmpdir, 'he.col', he_col)
    elem, xyz, vec, com = fileio.read_auto(f, hascom=True)
    assert np.all(elem == ['He'])
    assert np.allclose(xyz, np.zeros((1, 3)))
    assert com == 'comment line'


def test_read_auto_traj_one_line_comment(tmpdir):
    f = tmpf(tmpdir, 'ch4.tj', traj_com)
    elem, xyz, vec, com = fileio.read_auto(f, hascom=True)
    assert np.all(elem == 5*['X'])
    assert np.allclose(xyz, ch4[1], atol=1e-4)
    assert com == 'comment line'


def test_read_auto_unrecognized_one_line_comment(tmpdir):
    f = tmpf(tmpdir, 'unrec.xyz', 'comment line\n123 abc 456 def\n')
    with pytest.raises(IOError, match=r'single line input in unrecognized .*'):
        elem, xyz, vec, com = fileio.read_auto(f, hascom=True)


def test_read_auto_xyz(tmpdir):
    pass


def test_read_auto_col(tmpdir):
    pass


def test_read_auto_col_comment(tmpdir):
    pass


def test_read_auto_gdat(tmpdir):
    pass


def test_read_auto_zmt(tmpdir):
    pass


def test_read_auto_zmt_comment(tmpdir):
    pass


def test_read_auto_traj(tmpdir):
    pass


def test_read_auto_unrecognized_multi_lines(tmpdir):
    pass


def test_write_xyz_vec(tmpdir):
    pass


def test_write_xyz_bohr(tmpdir):
    pass


def test_write_col_no_comment(tmpdir):
    pass


def test_write_gdat_vec(tmpdir):
    pass


def test_write_zmt_comment(tmpdir):
    pass


def test_write_zmtvar_comment(tmpdir):
    pass


def test_write_auto_col(tmpdir):
    pass


def test_write_auto_gdat(tmpdir):
    pass


def test_write_auto_zmt(tmpdir):
    pass


def test_write_auto_zmtvar(tmpdir):
    pass


def test_write_auto_xyz(tmpdir):
    pass


def test_convert_xyz_to_col(tmpdir):
    pass


def test_convert_xyz_to_zmt_inunits_bohr(tmpdir):
    pass


def test_convert_xyz_to_gdat_outunits_angstrom(tmpdir):
    pass


def test_convert_xyz_to_zmtvar_hascom(tmpdir):
    pass


def test_convert_xyz_to_traj_hasvec(tmpdir):
    pass
