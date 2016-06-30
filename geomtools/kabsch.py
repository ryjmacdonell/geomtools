'''
Mapping of FMS spawn geometries onto MECI geometries

Input/Output
------------
Config file 'kabsch_cfg.py' must be located in the working directory. Takes a set of 
references 'MECI/ref.i.xyz' with i set by 'ref_ind' and a set of FMS spawn geometries
'seed.j/Spawn.k' with spawn types set by 'spawn_type', then writes geometries that 
map onto reference i to 'spawntype.i.xyz' with corresponding momentum vectors written
to 'momentum.i'.

Notes
-----
Mapping performed by the Kabsch algorithm (see en.wikipedia.org/wiki/Kabsch_algorithm).
Optimal mapping chosen from the minimum RMSD value of spawns vs. references.  Can give 
one or more sets of permutable atoms as 'pmute' by (cardinal) index. By default, 
inversions of each permutation are also tested against the reference (set by 'chiral').
Currently raises RuntimeWarning if no spawns exist of the given type.
'''
import numpy as np
from glob import glob
import os
import sys
sys.path.append(os.getcwd())
sys.dont_write_bytecode = True

# constants and conversion factors
a0 = 0.52917721092
atmnum = {'H':1, 'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, \
    'F':9, 'Ne':10, 'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, \
    'S':16, 'Cl':17, 'Ar':18}
atmmass = {'H':1.00782504, 'He':4.00260325, 'Li':7.01600450, 'Be':9.01218250, \
    'B':11.00930530, 'C':12.00000000, 'N':14.00307401, 'O':15.99491464, \
    'F':18.99840325, 'Ne':19.99243910, 'Na':22.98976970, 'Mg':23.98504500, \
    'Al':26.98154130, 'Si':27.97692840, 'P':30.97376340, 'S':31.97207180, \
    'Cl':34.96885273, 'Ar':39.96238310}

def read_cfg(req, opt):
    # read in variables from config file
    cfgvars = dir(cfg)

    for v in req:
        if v not in cfgvars:
            raise AttributeError('\'{}\' not found in kabsch_cfg.py'.format(v))

    for v in opt:
        if v not in cfgvars:
            setattr(cfg, v, globals()[v])

def read_ref(fname, v=False):
    # read reference geometry in xyz format
    if v:
        print 'Reading reference file {}...'.format(fname)
    inpf = open(fname, 'r')

    n = int(inpf.readline().split()[0])
    inpf.readline()

    p = np.chararray(n, itemsize=2)
    m = np.empty(n)
    xyz = np.empty((n, 3))
    for i in range(n):
        data = inpf.readline().split()
        p[i] = data[0]
        m[i] = atmmass[data[0]]

        xyz[i] = np.array(data[1:]).astype(float)

    inpf.close()

    return xyz, n, p, m

def read_spawn(fname, n, sp_type, v=False):
    # read spawn geometry in xyz format
    if v:
        print 'Reading file {}...'.format(fname)

    inpf = open(fname, 'r')

    inpf.readline()
    info = inpf.readline().split()

    t = float(info[0])
    s_i = int(info[9])
    s_f = int(info[3])
    parent = int(info[6])

    right_spawn = (s_i == sp_type[0] and s_f == sp_type[1])

    xyz = np.empty((n, 3))
    if right_spawn:
        for i in range(n):
            data = inpf.readline().split()
            xyz[i] = np.array(data[1:]).astype(float)

    inpf.close()

    return right_spawn, xyz, t, parent

def write_xyz(outf, n, p, xyz, note='', v=False):
    # write geometry to xyz format
    if v:
        print 'Writing to file {}...'.format(outf.name)

    outf.write(' {}\n{}\n'.format(n, note))

    for i in range(n):
        outf.write('     {:>2s}{:16f}{:16f}{:16f}\n'.format(p[i], *xyz[i]))

def write_mom(outf, n, xyz, note='', v=False):
    # write cartesian momentum vector
    if v:
        print 'Writing to file {}...'.format(outf.name)

    outf.write('{}\n'.format(note))

    for i in range(n):
        outf.write('{:15.6E}{:15.6E}{:15.6E}\n'.format(*xyz[i]))

def write_stats(sp_type, inds, totals, v=False):
    # write statistics for mapped spawns
    popf = open('MECI/spawns{}s{}/percentpop'.format(sp_type[0] - 1, sp_type[1] - 1), 'w')

    if v:
        print 'Writing to file {}...'.format(popf.name)

    popf.write('S{} to S{} MECI analysis\n'.format(sp_type[0] - 1, sp_type[1] - 1))
    popf.write('   total')
    popf.write(''.join(['  type.{}/%'.format(lbl) for lbl in inds]) + '\n')
    popf.write('{:8.4f}'.format(np.sum(totals)))
    popf.write(''.join(['{:10.4f}'.format(100 * totals[i] / np.sum(totals)) for i in \
        range(len(inds))]) + '\n')

    popf.close()

def centre_mass(xyz, m):
    # return geometry with centre of mass subtracted out
    xyz_new = np.copy(xyz)
    xyz_new -= np.sum(xyz * m[:, np.newaxis], axis=0) / np.sum(m)

    return xyz_new

def kabsch(P, Q):
    # perform Kabsch algorithm from geometries P (test) and Q (reference)
    A = P.T.dot(Q)
    V, S, W = np.linalg.svd(A)

    d = np.sign(np.linalg.det(V) * np.linalg.det(W))
    V[:, -1] *= d

    U = V.dot(W)

    return P.dot(U)

def rmsd(P, Q, N):
    # compute the RMSD of geometries P and Q
    return np.sqrt(np.sum((P - Q) ** 2) / N)

def tuple2list(tupl):
    # convert nested tuple to nested list
    return list((tuple2list(x) if isinstance(x, tuple) else x for x in tupl))

def perm(q):
    # generate an array of permutations of a list of lists of permutable indices
    import itertools

    if not isinstance(q[0], list):
        q = [q]

    orig = sum(q, [])
    
    sep_perms = [list(itertools.permutations(i)) for i in q]
    comb_perms = list(itertools.product(*sep_perms))
    comb_perms = tuple2list(comb_perms)
    comb_perms = [sum(i, []) for i in comb_perms]

    return orig, comb_perms

def select_singleref(test, ref, q=[[0]], chiral=False):
    # compute optimal mapping of test and its permutations/reflections onto ref
    geoms = []

    origin, perms = perm(q)
    for ind in perms:
        xyz = np.copy(test)
        xyz[origin] = xyz[ind]
        geoms.append(kabsch(xyz, ref))

        if not chiral:
            geoms.append(kabsch(-xyz, ref))

    rms = np.array([rmsd(g, ref, 3 * n) for g in geoms])

    return geoms[np.argmin(rms)], np.min(rms)

def select_multiref(test, refs, q=[[0]], chiral=False, v=False):
    # compute optimal ref for a given test geometry
    geoms = [[]] * len(refs)
    rms = np.empty(len(refs))

    for i in range(len(refs)):
        geoms[i], rms[i] = select_singleref(test, refs[i], q, chiral)

    if v:
        print 'RMSD = {}'.format(rms)

    return geoms[np.argmin(rms)], refs[np.argmin(rms)], np.argmin(rms)

def get_spawnpop(fname, n_pts=100):
    # read population spawned based on slope of pop vs. time
    f = open(fname, 'r')
    f.readline()

    data = np.empty(n_pts) * np.nan
    time = np.empty(n_pts) * np.nan
    slope = np.zeros(n_pts)

    try:
        for i in range(n_pts):
            split = f.readline().split()

            time[i] = split[0]
            data[i] = split[-2]

            if i > 0:
                slope[i-1] = (data[i] - data[i-1])/(time[i] - time[i-1])
                if i > 1 and slope[i-1] <= slope[i-2] and slope[i-1] < max(slope) / 25: 
                    break
    except IndexError:
        pass

    return data[i]

def get_spawnmom(fname, t_spawn, n):
    # get momentum of parent at time of spawn
    f = open(fname, 'r')
    f.readline()

    mom = np.empty(3 * n)

    for line in f:
        split = line.split()

        t = float(split[0])

        if t - t_spawn < 1:
            mom = np.array(split[3 * n + 1:-5]).astype(float)
            break

    return mom.reshape(n, 3)

def int_test(s):
    # change string to int for sorting
    return int(s) if s.isdigit() else s

def natural_keys(s):
    # key for sorting based on integer order
    import re
    return [int_test(c) for c in re.split('(\d+)', s)]

def clean_up(filelist, v=False):
    # remove empty output files
    for f in filelist:
        f.seek(0)
        empty = not bool(f.read())
        fname = f.name
        f.close()

        if empty:
            if v:
                print 'Removing empty file {}'.format(fname)

            os.remove(fname)
    

if __name__ == '__main__':
    # defaults for optional variables
    pmute = [[0]]
    chiral = False
    verbose = False

    # read in config data
    import kabsch_cfg as cfg

    reqvars = ['spawn_type', 'ref_ind']
    optvars = ['pmute', 'chiral', 'verbose']

    read_cfg(reqvars, optvars)

    # begin reading data files
    spawn_list = glob('seed.*/Spawn.[0-9]*')
    spawn_list.sort(key=natural_keys)

    xyz_ref = [[]] * len(cfg.ref_ind)
    outf = [[]] * len(cfg.ref_ind)
    momf = [[]] * len(cfg.ref_ind)
    total_pop = np.zeros(len(cfg.ref_ind))

    # read in reference data
    for i in range(len(cfg.ref_ind)):
        xyz_i, n, p, m = read_ref('MECI/ref.{}.xyz'.format(cfg.ref_ind[i]), v=cfg.verbose)
        xyz_ref[i] = centre_mass(xyz_i, m)

        outf[i] = open('MECI/spawns{}s{}/spawntype.{}.xyz'.format(cfg.spawn_type[0] - 1, \
            cfg.spawn_type[1] - 1, cfg.ref_ind[i]), 'w+')
        momf[i] = open('MECI/spawns{}s{}/momentum.{}'.format(cfg.spawn_type[0] - 1, \
            cfg.spawn_type[1] - 1, cfg.ref_ind[i]), 'w+')

    # read in and compute spawn data
    for spawn in spawn_list:
        spawned, xyz_spawn, t, parent = read_spawn(spawn, n, cfg.spawn_type, v=cfg.verbose)

        if spawned:
            xyz_spawn = centre_mass(xyz_spawn, m)
            best_spawn, best_ref, ind = select_multiref(xyz_spawn, xyz_ref, cfg.pmute, \
                cfg.chiral, v=cfg.verbose)

            trajfile = spawn.replace('Spawn', 'TrajDump')
            spawn_pop = get_spawnpop(trajfile)
            total_pop[ind] += spawn_pop

            parentfile = trajfile.rsplit('.', 1)[0] + '.{}'.format(parent)
            spawn_mom = get_spawnmom(parentfile, t, n)

            write_xyz(outf[ind], n, p, best_spawn, \
                '{} at {:10.2f} from traj {:4d} with {:10.4f} pop'.format(spawn, t, parent, \
                spawn_pop), v=cfg.verbose)
            write_mom(momf[ind], n, spawn_mom, '{} momentum vector'.format(spawn), \
                v=cfg.verbose)
    
    # output spawn stats
    write_stats(cfg.spawn_type, cfg.ref_ind, total_pop, v=cfg.verbose)

    clean_up(outf + momf, v=cfg.verbose)

    print 'Exited successfully.'
