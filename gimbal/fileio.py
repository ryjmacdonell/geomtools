"""
File input/output functions for molecular geometry files.

Can support XYZ, COLUMBUS and Z-matrix formats. Input and output both
require an open file to support multiple geometries. Only 3D geometries
are currently supported.
"""
import numpy as np
import gimbal.constants as con
import gimbal.displace as displace
import gimbal.measure as measure


def read_xyz(infile, units='ang', hasvec=False, hascom=False):
    """Reads input file in XYZ format.

    XYZ files are in the format::

        natm
        comment
        A X1 Y1 Z1 [Vx1 Vy1 Vz1]
        B X2 Y2 Z2 [Vx2 Vy2 Vz2]
        ...

    where natm is the number of atoms, comment is a comment line, A and B
    are atomic labels and X, Y and Z are cartesian coordinates (in
    Angstroms). The vectors Vx, Vy and Vz are optional and will only be
    read if hasvec = True.

    Due to the preceding number of atoms, multiple XYZ format geometries
    can easily be read from a single file.

    Parameters
    ----------
    infile : file
        The open input file.
    units : str, optional
        The units of length of the cartesian coordinates. Default is
        Angstroms.
    hasvec : bool, optional
        Specifies if a vector should be read from the input file. If
        False (default), vec is a zero array.
    hascom : bool, optional
        Specifies if a comment line should be read from the input file. If
        False (default), comment is a blank string.

    Returns
    -------
    elem : (N,) ndarray
        The atomic symbols.
    xyz : (N, 3) ndarray
        The atomic cartesian coordinates.
    vec : (N, 3) ndarray
        The vector cartesian coordinates.
    comment : str
        The comment line.

    Raises
    ------
    IOError
        When geometry is not in the correct format (natm can't be read).
    """
    try:
        natm = int(infile.readline())
    except ValueError:
        raise IOError('geometry not in XYZ format.')
    if hascom:
        comment = infile.readline().strip()
    else:
        infile.readline()
        comment = ''
    data = np.array([infile.readline().split() for i in range(natm)])
    elem = data[:, 0]
    xyz = data[:, 1:4].astype(float) * con.conv(units, 'ang')
    if hasvec:
        vec = data[:, 4:7].astype(float)
        if vec.size == 0:
            vec = np.zeros_like(xyz)
    else:
        vec = None
    return elem, xyz, vec, comment


def read_col(infile, units='bohr', hasvec=False, hascom=False):
    """Reads input file in COLUMBUS format.

    COLUMBUS geometry files are in the format::

        A nA X1 Y1 Z1 mA
        B nB X2 Y2 Z2 mB
        ...

    where A and B are atomic labels, nA and nB are corresponding atomic
    numbers, mA and mB are corresponding atomic masses and X, Y and Z
    are cartesian coordinates (in Bohrs).

    COLUMBUS geometry files do not provide the number of atoms in each
    geometry. A comment line (or blank line) must be used to separate
    molecules.

    For the time being, vector input is not supported for the
    COLUMBUS file format.

    Parameters
    ----------
    infile : file
        The open input file.
    units : str, optional
        The units of length of the cartesian coordinates. Default is
        Bohr.
    hasvec : bool, optional
        Specifies if a vector should be read from the input file. If
        False (default), vec is a zero array.
    hascom : bool, optional
        Specifies if a comment line should be read from the input file. If
        False (default), comment is a blank string.

    Returns
    -------
    elem : (N,) ndarray
        The atomic symbols.
    xyz : (N, 3) ndarray
        The atomic cartesian coordinates.
    vec : (N, 3) ndarray
        The vector cartesian coordinates.
    comment : str
        The comment line.
    """
    if hascom:
        comment = infile.readline().strip()
    else:
        comment = ''
    data = np.empty((0, 6), dtype=str)
    while True:
        pos = infile.tell()
        line = np.array(infile.readline().split())
        try:
            # catch comment line or end-of-file
            line[1:].astype(float)
            data = np.vstack((data, line))
        except (ValueError, IndexError):
            if len(data) < 1:
                raise IOError('geometry not in COLUMBUS format.')
            else:
                # roll back one line before break
                infile.seek(pos)
                break
    elem = data[:, 0]
    xyz = data[:, 2:5].astype(float) * con.conv(units, 'ang')
    if hasvec:
        vec = np.zeros_like(xyz)
    else:
        vec = None
    return elem, xyz, vec, comment


def read_gdat(infile, units='bohr', hasvec=False, hascom=False):
    """Reads input file in FMS90 Geometry.dat format.

    Geometry.dat files are in the format::

        comment
        natm
        A X1 Y1 Z1
        B X2 Y2 Z2
        ...
        Vx1 Vy1 Vz1
        Vx2 Vy2 Vz2
        ...

    where comment is a comment line, natm is the number of atoms, A and B
    are atomic labels, X, Y and Z are cartesian coordinates and Vq are
    vectors for cartesian coordinates q. The vectors are only read if
    hasvec = True.

    Parameters
    ----------
    infile : file
        The open input file.
    units : str, optional
        The units of length of the cartesian coordinates. Default is
        Bohr.
    hasvec : bool, optional
        Specifies if a vector should be read from the input file. If
        False (default), vec is a zero array.
    hascom : bool, optional
        Specifies if a comment line should be read from the input file. If
        False (default), comment is a blank string.

    Returns
    -------
    elem : (N,) ndarray
        The atomic symbols.
    xyz : (N, 3) ndarray
        The atomic cartesian coordinates.
    vec : (N, 3) ndarray
        The vector cartesian coordinates.
    comment : str
        The comment line.

    Raises
    ------
    IOError
        When geometry is not in the correct format (natm can't be read).
    """
    if hascom:
        comment = infile.readline().strip()
    else:
        infile.readline()
        comment = ''
    try:
        natm = int(infile.readline())
    except ValueError:
        raise IOError('geometry not in Geometry.dat format')
    data = np.array([infile.readline().split() for i in range(natm)])
    elem = data[:, 0]
    xyz = data[:, 1:].astype(float) * con.conv(units, 'ang')
    if hasvec:
        vec = np.array([infile.readline().split() for i in range(natm)],
                       dtype=float)
    else:
        vec = None
    return elem, xyz, vec, comment


def read_zmt(infile, units='ang', hasvec=False, hascom=False):
    """Reads input file in Z-matrix format.

    Z-matrix files are in the format::

        A
        B 1 R1
        C indR2 R2 indA2 A2
        D indR3 R3 indA3 A3 indT3 T3
        E indR4 R4 indA4 A4 indT4 T4
        ...

    where A, B, C, D, E are atomic labels, indR, indA, indT are reference
    atom indices, R are bond lengths (in Angstroms), A are bond angles (in
    degrees) and T are dihedral angles (in degrees). For example, E is a
    distance R from atom indR with an E-indR-indA angle of A and an
    E-indR-indA-indT dihedral angle of T. Alternatively, values can be
    assigned to a list of variables after the Z-matrix (preceded by a blank
    line).

    Although the number of atoms is not provided, the unique format of the
    first atom allows multiple geometries to be read without separation
    by a comment line.

    For the time being, vector input is not supported for the
    Z-matrix file format.

    Parameters
    ----------
    infile : file
        The open input file.
    units : str, optional
        The units of length of the cartesian coordinates. Default is
        Angstroms.
    hasvec : bool, optional
        Specifies if a vector should be read from the input file. If
        False (default), vec is a zero array.
    hascom : bool, optional
        Specifies if a comment line should be read from the input file. If
        False (default), comment is a blank string.

    Returns
    -------
    elem : (N,) ndarray
        The atomic symbols.
    xyz : (N, 3) ndarray
        The atomic cartesian coordinates.
    vec : (N, 3) ndarray
        The vector cartesian coordinates.
    comment : str
        The comment line.

    Raises
    ------
    IOError
        When geometry is not in the correct format (no atoms found).
    """
    if hascom:
        comment = infile.readline().strip()
    else:
        comment = ''
    data = []
    vlist = dict()
    while True:
        pos = infile.tell()
        line = infile.readline()
        split = line.split()
        if line == '':
            # end-of-file
            break
        elif len(split) == 1 and len(data) > 0:
            # roll back one line before break
            infile.seek(pos)
            break
        elif split == []:
            # blank line before variable assignment
            continue
        elif split[0] in con.sym:
            data.append(split)
        elif split[1] == '=' and len(split) == 3:
            vlist[split[0]] = float(split[2])
        else:
            # assume it's a comment line and roll back
            infile.seek(pos)
            break

    natm = len(data)
    if natm < 1:
        raise IOError('geometry not in Z-matrix format.')
    elem = np.array([line[0] for line in data])
    xyz = np.zeros((natm, 3))
    for i in range(natm):
        if i == 0:
            # leave first molecule at origin
            continue
        elif i == 1:
            # move along z-axis by R
            xyz = displace.translate(xyz, _valvar(data[1][2], vlist),
                                     [0, 0, 1], ind=1)
        elif i == 2:
            indR = int(data[2][1]) - 1
            indA = int(data[2][3]) - 1
            xyz[2] = xyz[indR]
            # move from indR towards indA by R
            xyz = displace.translate(xyz, _valvar(data[2][2], vlist),
                                     xyz[indA]-xyz[indR], ind=2)
            # rotate into xz-plane by A
            xyz = displace.rotate(xyz, _valvar(data[2][4], vlist), [0, 1, 0],
                                  ind=2, origin=xyz[indR], units='deg')
        else:
            indR = int(data[i][1]) - 1
            indA = int(data[i][3]) - 1
            indT = int(data[i][5]) - 1
            xyz[i] = xyz[indR]
            # move from indR towards indA by R
            xyz = displace.translate(xyz, _valvar(data[i][2], vlist),
                                     xyz[indA]-xyz[indR], ind=i)
            # rotate about (indT-indA)x(indR-indA) by A
            xyz = displace.rotate(xyz, _valvar(data[i][4], vlist),
                                  np.cross(xyz[indT]-xyz[indA],
                                           xyz[indR]-xyz[indA]),
                                  ind=i, origin=xyz[indR], units='deg')
            # rotate about indR-indA by T
            xyz = displace.rotate(xyz, _valvar(data[i][6], vlist),
                                  xyz[indR]-xyz[indA],
                                  ind=i, origin=xyz[indR], units='deg')

    xyz = displace.centre_mass(elem, xyz) * con.conv(units, 'ang')
    if hasvec:
        vec = np.zeros_like(xyz)
    else:
        vec = None
    return elem, xyz, vec, comment


def read_traj(infile, units='bohr', hasvec=False, hascom=False,
              elem=None, time=None, autocom=False):
    """Reads input file in FMS/nomad trajectory format

    trajectory files are in the format::

        T1 X1 Y1 Z1 X2 Y2 ... Vx1 Vy1 Vz1 Vx2 Vy2 ... G Re(A) Im(A) |A| S
        T2 X1 Y1 Z1 X2 Y2 ... Vx1 Vy1 Vz1 Vx2 Vy2 ... G Re(A) Im(A) |A| S
        ...

    where T is the time, Vq are the vectors (momenta) for cartesian
    coordinates q, G is the phase, A is the amplitude and S is the state
    label. The vectors are only read if hasvec = True.

    Trajectory files do not contain atomic labels. If not provided, they are
    set to dummy atoms which may affect calculations involving atomic
    properties. A time should be provided, otherwise the first geometry
    in the file is used.

    Parameters
    ----------
    infile : file
        The open input file.
    units : str, optional
        The units of length of the cartesian coordinates. Default is
        Bohr.
    hasvec : bool, optional
        Specifies if a vector should be read from the input file. If
        False (default), vec is a zero array.
    hascom : bool, optional
        Specifies if a comment line should be read from the input file. If
        False (default), comment is a blank string.
    elem : (N,) array_like, optional
        A list of the atomic symbols. If elem is None (default), the
        symbols will all be set to 'X'.
    time : float, optional
        The desired time of the trajectory. If time is None (default), the
        first geometry in the buffer is parsed. Otherwise, the file is
        read until the specified time is found.
    autocom : bool, optional
        Specifies if a comment line should be automatically generated with
        time, state and squared amplitude. Default is False.

    Returns
    -------
    elem : (N,) ndarray
        The atomic symbols.
    xyz : (N, 3) ndarray
        The atomic cartesian coordinates.
    vec : (N, 3) ndarray
        The vector cartesian coordinates.
    comment : str
        The comment line.

    Raises
    ------
    IOError
        When the geometry is not in the correct format (empty line or
        incorrect number of columns).
    """
    if hascom:
        comment = infile.readline().strip()
    else:
        comment = ''
    if time is None:
        rawline = infile.readline().split()
        if rawline == []:
            raise IOError('empty line provided')
        elif 'Time' in rawline:
            line = np.array(infile.readline().split(), dtype=float)
        else:
            line = np.array(rawline, dtype=float)
    else:
        alldata = np.array([line.split() for line in infile.readlines()
                            if 'Time' not in line], dtype=float)
        line = alldata[np.isclose(alldata[:,0], time)][0]
    natm = len(line) // 6 - 1
    if natm < 1 or len(line) % 6 != 0:
        raise IOError('geometry not in trajectory format.')
    if elem is None:
        elem = np.array(['X'] * natm)
    xyz = line[1:3*natm+1].reshape(natm, 3) * con.conv(units,'ang')
    if hasvec:
        vec = line[3*natm+1:6*natm+1].reshape(natm, 3)
    else:
        vec = None
    if autocom:
        fmt = 't={:8.2f}, state={:4d}, a^2={:10.4f}'
        comment += fmt.format(line[0], int(line[-1]), line[-2])
    return elem, xyz, vec, comment


def read_auto(infile, hasvec=False, hascom=False, **kwargs):
    """Reads a molecular geometry file and determines the format.

    Parameters
    ----------
    infile : file
        The open input file.
    hasvec : bool, optional
        Specifies if a vector should be read from the input file.
    hascom : bool, optional
        Specifies if a comment line should be read from the input file.
    kwargs : dict, optional
        Additional keyword arguments for the read functions.

    Returns
    -------
    elem : (N,) ndarray
        The atomic symbols.
    xyz : (N, 3) ndarray
        The atomic cartesian coordinates.
    vec : (N, 3) ndarray
        The vector cartesian coordinates.
    comment : str
        The comment line.

    Raises
    ------
    IOError
        When the geometry format is not recognized by the input parser.
    """
    kwargs.update(dict(hascom=hascom, hasvec=hasvec))
    pos = infile.tell()
    contents = infile.readlines()
    infile.seek(pos)
    nlines = min(len(contents), 4)
    fmt = []
    for i in range(nlines):
        line = contents[i].split()
        fmt.append(''.join([_get_type(l) for l in line]))

    if nlines == 0:
        raise IOError('end of file')
    elif nlines == 1:
        if hascom:
            raise IOError('cannot have comment with single line file')
        else:
            if fmt[0] == 's':
                return read_zmt(infile, **kwargs)
            elif fmt[0].replace('i', 'f') == 'sfffff':
                return read_col(infile, **kwargs)
            elif 'ffffffffffff' in fmt[0].replace('i', 'f'):
                return read_traj(infile, **kwargs)
            else:
                raise IOError('single line input in unrecognized format')
    elif nlines == 2 and hascom:
        if fmt[1] == 's':
            return read_zmt(infile, **kwargs)
        elif fmt[1].replace('i', 'f') == 'sfffff':
            return read_col(infile, **kwargs)
        elif 'ffffffffffff' in fmt[1].replace('i', 'f'):
            return read_traj(infile, **kwargs)
        else:
            raise IOError('single line input in unrecognized format')
    else:
        if hascom:
            if fmt[1] == 's' and fmt[2] in ['sis', 'sif', 'sii']:
                return read_zmt(infile, **kwargs)
            elif [f.replace('i', 'f') for f in fmt[1:3]] == ['sfffff', 'sfffff']:
                return read_col(infile, **kwargs)
        else:
            if fmt[0] == 's' and fmt[1] in ['sis', 'sif', 'sii']:
                return read_zmt(infile, **kwargs)
            elif [f.replace('i', 'f') for f in fmt[:2]] == ['sfffff', 'sfffff']:
                return read_col(infile, **kwargs)

        if ('ffffffffffff' in fmt[0].replace('i', 'f') or
            'ffffffffffff' in fmt[1].replace('i', 'f')):
            return read_traj(infile, **kwargs)
        elif nlines > 2:
            if fmt[0] == 'i':
                return read_xyz(infile, **kwargs)
            elif fmt[1] == 'i':
                return read_gdat(infile, **kwargs)

    raise IOError('unrecognized file format')


def read_single(infile, fmt='auto', **kwargs):
    """Reads a single geometry from an input file.

    Unlike :func:`read_auto`, infile can be a string or open file.

    Parameters
    ----------
    infile : file or str
        The open input file or filename.
    fmt : str, optional
        The file format. Default is auto (i.e. :func:`read_auto`).
    kwargs : dict, optional
        Additional keyword arguments for the read function.

    Returns
    -------
    tuple
        The output of the read function (elem, xyz, vec, comment).
    """
    read_func = globals()['read_' + fmt]
    close = False
    infile, close = _open_file(infile, 'r')
    moldat = read_func(infile, **kwargs)
    if close:
        infile.close()

    return moldat


def read_multiple(inflist, fmt='auto', **kwargs):
    """Reads multiple files or multiple geometries into lists of data.

    Parameters
    ----------
    inflist : array_like
        The open input files or filenames.
    fmt : str, optional
        The file format. Default is auto (i.e. :func:`read_auto`).
    kwargs : dict, optional
        Additional keyword arguments for the read functions.

    Returns
    -------
    list
        The outputs of the read function for each file in inflist.

    Raises
    ------
    IOError
        When no geometries are found in the input file list.
    """
    read_func = globals()['read_' + fmt]
    inflist = np.atleast_1d(inflist)
    moldat = []
    for infile in inflist:
        infile, close = _open_file(infile, 'r')
        while True:
            try:
                moldat.append(read_func(infile, **kwargs))
            except IOError:
                break

        if close:
            infile.close()

    if len(moldat) == 0:
        raise IOError('no geometries read from input files')

    return moldat


def write_xyz(outfile, elem, xyz, vec=None, comment='', units='ang'):
    """Writes geometry to an output file in XYZ format.

    Parameters
    ----------
    outfile : file
        The open output file.
    elem : (N,) array_like
        The atomic symbols.
    xyz : (N, 3) array_like
        The atomic cartesian coordinates.
    vec : (N, 3) array_like, optional
        The atomic cartesian vectors. If vec is None (default), it is not
        written to the output file.
    comment : str, optional
        The comment line. Default is a blank string.
    units : str, optional
        The units of the cartesian coordinates. Default is Angstroms.
    """
    natm = len(elem)
    write_xyz = xyz * con.conv('ang', units)
    outfile.write(' {}\n{}\n'.format(natm, comment))
    if vec is None:
        for atm, xyzi in zip(elem, write_xyz):
            outfile.write('{:4s}{:12.6f}{:12.6f}{:12.6f}\n'.format(atm, *xyzi))
    else:
        for atm, xyzi, pxyzi in zip(elem, write_xyz, vec):
            outfile.write('{:4s}{:12.6f}{:12.6f}{:12.6f}'.format(atm, *xyzi) +
                          '{:12.6f}{:12.6f}{:12.6f}\n'.format(*pxyzi))


def write_col(outfile, elem, xyz, vec=None, comment='', units='bohr'):
    """Writes geometry to an output file in COLUMBUS format.

    For the time being, vector output is not supported for the
    COLUMBUS file format.

    Parameters
    ----------
    outfile : file
        The open output file.
    elem : (N,) array_like
        The atomic symbols.
    xyz : (N, 3) array_like
        The atomic cartesian coordinates.
    vec : (N, 3) array_like, optional
        The atomic cartesian vectors. If vec is None (default), it is not
        written to the output file.
    comment : str, optional
        The comment line. Default is a blank string.
    units : str, optional
        The units of the cartesian coordinates. Default is Bohr.
    """
    write_xyz = xyz * con.conv('ang', units)
    if comment != '':
        outfile.write(comment + '\n')
    for atm, (x, y, z) in zip(elem, write_xyz):
        outfile.write(' {:<2s}{:7.1f}{:14.8f}{:14.8f}{:14.8f}{:14.8f}'
                      '\n'.format(atm, con.get_num(atm), x, y, z,
                                  con.get_mass(atm)))


def write_gdat(outfile, elem, xyz, vec=None, comment='', units='bohr'):
    """Writes geometry to an output file in Geometry.dat format.

    Parameters
    ----------
    outfile : file
        The open output file.
    elem : (N,) array_like
        The atomic symbols.
    xyz : (N, 3) array_like
        The atomic cartesian coordinates.
    vec : (N, 3) array_like, optional
        The atomic cartesian vectors. If vec is None (default), the
        vectors are replaced by zeros.
    comment : str, optional
        The comment line. Default is a blank string.
    units : str, optional
        The units of the cartesian coordinates. Default is Bohr.
    """
    natm = len(elem)
    write_xyz = xyz * con.conv('ang', units)
    outfile.write('{}\n{}\n'.format(comment, natm))
    for atm, xyzi in zip(elem, write_xyz):
        outfile.write('{:<2s}{:18.8E}{:18.8E}{:18.8E}\n'.format(atm, *xyzi))
    if vec is None:
        for line in range(natm):
            outfile.write(' {:18.8E}{:18.8E}{:18.8E}\n'.format(0, 0, 0))
    else:
        for pxyzi in vec:
            outfile.write(' {:18.8E}{:18.8E}{:18.8E}\n'.format(*pxyzi))


def write_zmt(outfile, elem, xyz, vec=None, comment='', units='ang'):
    """Writes geometry to an output file in Z-matrix format.

    At present, each atom uses the previous atoms in order as
    references. This could be made 'smarter' using the bonding module.

    For the time being, vector output is not supported for the
    Z-matrix file format.

    Parameters
    ----------
    outfile : file
        The open output file.
    elem : (N,) array_like
        The atomic symbols.
    xyz : (N, 3) array_like
        The atomic cartesian coordinates.
    vec : (N, 3) array_like, optional
        The atomic cartesian vectors. If vec is None (default), it is not
        written to the output file.
    comment : str, optional
        The comment line. Default is a blank string.
    units : str, optional
        The units of the bond distances. Default is Angstroms.
    """
    natm = len(elem)
    if comment != '':
        outfile.write(comment + '\n')
    for i in range(natm):
        if i == 0:
            # first element has just the symbol
            outfile.write('{:s}\n'.format(elem[0]))
        elif i == 1:
            # second element has symbol, index, bond length
            outfile.write('{:<2s}{:3d}{:12.6f}'
                          '\n'.format(elem[1], 1, measure.stre(xyz, 0, 1,
                                                               units=units)))
        elif i == 2:
            # third element has symbol, index, bond length, index, bond angle
            outfile.write('{:<2s}{:3d}{:12.6f}{:3d}{:12.6f}'
                          '\n'.format(elem[2], 2, measure.stre(xyz, 1, 2,
                                                               units=units),
                                      1, measure.bend(xyz, 0, 1, 2,
                                                      units='deg')))
        else:
            # all other elements have symbol, index, bond length, index,
            # bond angle, index, dihedral angle
            outfile.write('{:<2s}{:3d}{:12.6f}{:3d}{:12.6f}{:3d}{:12.6f}'
                          '\n'.format(elem[i], i, measure.stre(xyz, i-1, i,
                                                               units=units),
                                      i-1, measure.bend(xyz, i-2, i-1, i,
                                                        units='deg'),
                                      i-2, measure.tors(xyz, i-3, i-2, i-1, i,
                                                        units='deg')))


def write_zmtvar(outfile, elem, xyz, vec=None, comment='', units='ang'):
    """Writes geometry to an output file in Z-matrix format with
    variable assignments.

    For the time being, vector output is not supported for the
    Z-matrix file format.

    Parameters
    ----------
    outfile : file
        The open output file.
    elem : (N,) array_like
        The atomic symbols.
    xyz : (N, 3) array_like
        The atomic cartesian coordinates.
    vec : (N, 3) array_like, optional
        The atomic cartesian vectors. If vec is None (default), it is not
        written to the output file.
    comment : str, optional
        The comment line. Default is a blank string.
    units : str, optional
        The units of the bond distances. Default is Angstroms.
    """
    natm = len(elem)
    if comment != '':
        outfile.write(comment + '\n')
    vlist = []
    vdict = dict()
    for i in range(natm):
        if i == 0:
            # first element has just the symbol
            outfile.write('{:s}\n'.format(elem[0]))
        elif i == 1:
            # second element has symbol, index, bond length
            outfile.write('{:<2s}{:3d}  R{:d}'
                          '\n'.format(elem[1], 1, 1))
            vlist += ['R1']
            vdict['R1'] = measure.stre(xyz, 0, 1, units=units)
        elif i == 2:
            # third element has symbol, index, bond length, index, bond angle
            outfile.write('{:<2s}{:3d}  R{:<2d} {:3d}  A{:d}'
                          '\n'.format(elem[2], 2, 2, 1, 1))
            vlist += ['R2', 'A1']
            vdict['R2'] = measure.stre(xyz, 1, 2, units=units)
            vdict['A1'] = measure.bend(xyz, 0, 1, 2, units='deg')
        else:
            # all other elements have symbol, index, bond length, index,
            # bond angle, index, dihedral angle
            outfile.write('{:<2s}{:3d}  R{:<2d} {:3d}  A{:<2d} '
                          '{:3d}  T{:d}'
                          '\n'.format(elem[i], i, i, i-1, i-1, i-2, i-2))
            vlist += ['R'+str(i), 'A'+str(i-1), 'T'+str(i-2)]
            vdict['R'+str(i)] = measure.stre(xyz, i-1, i, units=units)
            vdict['A'+str(i-1)] = measure.bend(xyz, i-2, i-1, i, units='deg')
            vdict['T'+str(i-2)] = measure.tors(xyz, i-3, i-2, i-1, i,
                                               units='deg')
    outfile.write('\n')
    for key in vlist:
        outfile.write('{:4s} = {:14.8f}\n'.format(key, vdict[key]))


def write_traj(outfile, elem, xyz, vec=None, comment='', units='bohr',
               time=0., phase=0., ramp=0., iamp=0., state=0.):
    """Writes geometry to an output file in FMS/nomad trajectory format.

    Parameters
    ----------
    outfile : file
        The open output file.
    elem : (N,) array_like
        The atomic symbols.
    xyz : (N, 3) array_like
        The atomic cartesian coordinates.
    vec : (N, 3) array_like, optional
        The atomic cartesian vectors. If vec is None (default), the
        vectors are written as zeros.
    comment : str, optional
        The comment line. Default is a blank string.
    units : str, optional
        The units of the cartesian coordinates. Default is Bohr.
    time : float, optional
        The time of the trajectory. Default is zero.
    phase : float, optional
        The phase of the trajectory. Default is zero.
    ramp : float, optional
        The real amplitude of the trajectory. Default is zero.
    iamp : float, optional
        The imaginary amplitude of the trajectory. Default is zero.
    state : float, optional
        The state of the trajectory. Default is zero.
    """
    natm = len(elem)
    write_xyz = xyz.flatten() * con.conv('ang', units)
    if comment != '':
        outfile.write(comment + '\n')
    if vec is None:
        write_vec = np.zeros_like(write_xyz)
    else:
        write_vec = vec.flatten()
    namp = ramp**2 + iamp**2
    args = np.hstack((write_xyz, write_vec, phase, ramp, iamp, namp, state))
    fmt = '{:10.2f}' + (6*natm + 5)*'{:10.4f}' + '\n'
    outfile.write(fmt.format(time, *args))


def write_auto(outfile, elem, xyz, vec=None, comment='', **kwargs):
    """Writes geometry to an output file based on the filename extension.

    Extensions are not case sensitive. If the extension is not recognized,
    the default format is XYZ.

    Parameters
    ----------
    outfile : file
        The open output file.
    elem : (N,) array_like
        The atomic symbols.
    xyz : (N, 3) array_like
        The atomic cartesian coordinates.
    vec : (N, 3) array_like, optional
        The atomic cartesian vectors. Default is None.
    comment : str, optional
        The comment line. Default is a blank string.
    kwargs : dict, optional
        Additional keyword arguments for the write functions.
    """
    kwargs.update(dict(vec=vec, comment=comment))
    fname = outfile.name.lower()
    ext = fname.split('.')[-1]
    if ext in ['col', 'columbus']:
        write_col(outfile, elem, xyz, **kwargs)
    elif ext == 'dat':
        write_gdat(outfile, elem, xyz, **kwargs)
    elif ext in ['zmt', 'zmat', 'zmatrix']:
        write_zmt(outfile, elem, xyz, **kwargs)
    elif ext in ['zmtvar', 'zmatvar']:
        write_zmtvar(outfile, elem, xyz, **kwargs)
    elif ext in ['tj', 'traj']:
        write_traj(outfile, elem, xyz, **kwargs)
    else:
        write_xyz(outfile, elem, xyz, **kwargs)


def write_single(outfile, moldat, fmt='auto', **kwargs):
    """Writes a single geometry to an output file.

    Unlike :func:`write_auto`, outfile can be a string or open file.

    Parameters
    ----------
    outfile : file or str
        The open output file or filename.
    moldat : tuple
        The inputs of the write function.
    fmt : str, optional
        The file format. Default is auto (i.e. :func:`write_auto`).
    kwargs : dict, optional
        Additional keyword arguments for the write functions.
    """
    write_func = globals()['write_' + fmt]
    outfile, close = _open_file(outfile, 'w')
    write_func(outfile, *moldat, **kwargs)
    if close:
        outfile.close()


def write_multiple(outfile, moldat, fmt='auto', **kwargs):
    """Writes multiple geometries into a single output file.

    Parameters
    ----------
    outfile : file or str
        The open output file or filename.
    moldat : list
        A list of the inputs of the write function for each molecule.
    fmt : str, optional
        The file format. Default is auto (i.e. :func:`write_auto`).
    kwargs : dict, optional
        Additional keyword arguments for the write functions.
    """
    moldat = np.atleast_1d(moldat)
    write_func = globals()['write_' + fmt]
    outfile, close = _open_file(outfile, 'w')
    for dat in moldat:
        write_func(outfile, *dat, **kwargs)

    if close:
        outfile.close()


def convert(inflist, outfile, infmt='auto', outfmt='auto',
            inunits=None, outunits=None, hasvec=False, hascom=False):
    """Reads a file in format infmt and writes to a file in format
    outfmt.

    Input (and output) may have multiple geometries. Z-matrix index
    ordering is not conserved.

    Parameters
    ----------
    inflist : array_like
        The open input files or filenames.
    outfile : file or str
        The open output file or filename.
    infmt : str, optional
        The input file format. Default is auto (i.e. :func:`read_auto`).
    outfmt : str, optional
        The input file format. Default is auto (i.e. :func:`write_auto`).
    inunits : str, optional
        The input distance units. If None (default), the default units
        of the read function are used.
    outunits : str, optional
        The output distance units. If None (default), the default units
        of the write function are used.
    hasvec : bool, optional
        Specifies if a vector should be read from the input file.
    hascom : bool, optional
        Specifies if a comment line should be read from the input file.
    """
    inpkw = dict(hasvec=hasvec, hascom=hascom)
    outkw = dict()
    if inunits is not None:
        inpkw.update(units=inunits)
    if outunits is not None:
        outkw.update(units=outunits)

    moldat = read_multiple(inflist, fmt=infmt, **inpkw)
    write_multiple(outfile, moldat, fmt=outfmt, **outkw)


def get_optarg(arglist, *opts, default=False):
    """Gets an optional command line argument and returns its value.

    If default is not set, the flag is treated as boolean. Note that
    that setting default to None or '' will still take an argument
    after the flag.

    Parameters
    ----------
    arglist : array_like
        The command line argument list to be parsed.
    opts : list
        The arguments searched for in arglist.
    default : str or bool
        The default value if opts are not found in arglist. If default
        is False (default), then True is returned if opts are found.

    Returns
    -------
    str or bool
        The argument value in arglist or its default value.
    """
    for op in opts:
        if op in arglist:
            ind = arglist.index(op)
            arglist.remove(op)
            if default is False:
                return True
            else:
                return arglist.pop(ind)

    return default


def _get_type(s):
    """Reads a string to see if it can be converted into int or float.

    Parameters
    ----------
    s : str
        A string to be parsed.

    Returns
    -------
    str, int or float
        The parsed value.
    """
    try:
        float(s)
        if '.' not in s:
            return 'i'
        else:
            return 'f'
    except ValueError:
        return 's'


def _valvar(unk, vardict):
    """Determines if an unknown string is a value or a dict variable.

    Parameters
    ----------
    unk : float or str
        The unknown value, either a float or a dictionary key.
    vardict : dict
        The dictionary to be searched if unk is not a float.

    Returns
    -------
    float
        The desired value for unk.

    Raises
    ------
    ValueError
        When unk is not a float and not a key in vardict.
    """
    try:
        return float(unk)
    except ValueError:
        if unk in vardict:
            return vardict[unk]
        else:
            raise KeyError('\'{}\' not found in variable list'.format(unk))


def _open_file(f, mode):
    """Opens a file if a string is provided, otherwise leaves the file open.

    Parameters
    ----------
    f : str or file
        The open file or filename.
    mode : str
        The file mode for builtin function `open` (r, w, ...).

    Returns
    -------
    file
        The desired open file.
    """
    if isinstance(f, str):
        return open(f, mode), True
    else:
        return f, False
