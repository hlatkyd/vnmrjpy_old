#!/usr/local/bin/python3.6

import sys
import os
sys.path.append('/home/david/dev/vnmrjpy')
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cshelper as csh
import pywt

TESTDIR = '/home/david/dev/vnmrjpy/tempdata/cstest/ge3d_angio_HD_s_2018072604_HD_01.cs'

#sparse axis = 1, 2

class CompressedSensingRecon():

    def __init__(self, data):
    
        pass

kspace_real = []
kspace_imag = []
mask = []
for i in sorted(glob.glob(TESTDIR+'/kspace_real*')):
    print(i)
    kspace_real.append(nib.load(i).get_fdata())
for i in sorted(glob.glob(TESTDIR+'/kspace_imag*')):
    print(i)
    kspace_imag.append(nib.load(i).get_fdata())
for i in sorted(glob.glob(TESTDIR+'/kspace_mask*')):
    mask.append(nib.load(i).get_fdata())

data_real = np.asarray(kspace_real)
data_imag = np.asarray(kspace_imag)
mask = np.expand_dims(mask[0],axis=0)
print('mask shape: {}'.format(mask.shape))
#plt.imshow(mask[0,:,:,80,0])
#plt.show()
kspace = np.vectorize(complex)(data_real, data_imag)
kspace_cs = np.multiply(kspace, mask)

print(kspace_cs.shape)
plt.imshow(np.absolute(kspace_cs[0,:,:,100,0]))
plt.show()


