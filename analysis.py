import numpy as np
import matplotlib.pyplot as plt
import sys, time, os, datetime
import h5py as h5
import scipy as sci

dataDir = "/home/bruno/Desktop/data/qTurbulence/snapshots/"
inFileName = "psi_fwd_0.h5"

snapsFile = h5.File( dataDir + inFileName, 'r')

nSnap = 10
snapKey = 'snap_{0:03d}'.format(nSnap)
snapshot = snapsFile[snapKey]
time = snapshot.attrs['t']
psi = snapshot['psi'][...]
dens = np.abs( psi )**2


plt.imshow( dens[64,:,: ] )
plt.show()