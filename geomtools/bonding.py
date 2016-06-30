'''
Script for reading in atoms and determining their internal coordinates based on bonding.

Notes
-----
Coordinate determination not yet set up for rings. Rings are complicated.
'''
import numpy as np

def minor(arr, i, j)
    rows = np.array(range(i) + range(i + 1, arr.shape[0]))[:, np.newaxis]
    cols = np.array(range(j) + range(j + 1, arr.shape[1]))

    return arr[rows, cols]

atmsym = {'H':1, 'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, \
    'Ne':10, 'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18}
covrad = np.array([0.000, 0.320, 1.600, 0.680, 0.352, 0.832, 0.720, 0.680, 0.680, 0.640, \
    1.120, 0.972, 1.100, 1.352, 1.200, 1.036, 1.020, 1.000, 1.568])
error = 0.56

f = open('geom.xyz', 'r')
natm = int(f.readline())
f.readline()

data = f.readlines()
f.close()
atom = np.zeros(natm)
rad = np.zeros(natm)
xyz = np.zeros((natm, 3))

for i in range(natm):
    split = data[i].split()
    atom[i] = atmsym[split[0]]
    rad[i] = covrad[atom[i]]
    xyz[i] = np.array(split[1:]).astype(float)

xyz_diff = xyz.T[:,:,np.newaxis] - xyz.T[:,np.newaxis,:]
blength = np.sqrt(np.sum(xyz_diff ** 2, axis=0))

# build adjacency matrix from thresholds (cf. jmol, 0.35 from higher order bonding)
upthresh = np.add.outer(rad, rad) + error
lowthresh = upthresh - 0.35 - 2 * error
lowthresh[np.where(lowthresh < 0)] = 0

bonded = (blength < upthresh) & (blength > lowthresh)

# determine terminal atoms for path lengths of interest
b = bonded.astype(int)
c = b.dot(b) # oop angles can be determined using ij = 1 where ii > 2 (i != j)
c[np.eye(natm).astype(bool)] = 0
d = b.dot(b).dot(b)
d[(b + c).astype(bool)] = 0

print 'Length 1 paths (bond lengths):'
print b
print '\nLength 2 paths (bond angles):'
print c
print '\nLength 3 paths (torsional angles):'
print d

# find rings
eigs = np.linalg.eig(b)[0]
loop3 = np.sum(eigs ** 3) / 6
loop3 = int(round(loop3))
loop4 = (np.sum(eigs ** 4) - 2 * np.sum(b.dot(b)) + np.sum(b)) / 8
loop4 = int(round(loop4))

print '\nNumber of 3 membered rings: {}'.format(loop3)
print 'Number of 4 membered rings: {}'.format(loop4)
