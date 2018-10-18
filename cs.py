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

TESTDIR = '/home/david/dev/vnmrjpy_data/testdata/cstest/ge3d_angio_HD_s_2018072604_HD_01.cs'
SLICE = 100
SUBPLOT_ROW = 2
SUBPLOT_COL = 2
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
kspace = np.vectorize(complex)(data_real, data_imag)
"""
#TODO DC removal????
"""
kspace_cs = np.multiply(kspace, mask)
"""
Making original and zerofilled data
"""
imgspace = np.fft.ifftn(kspace, axes=(1,2,3))
imgspace = np.fft.fftshift(imgspace, axes=(1,2,3))
imgspace = np.sum(imgspace, axis=0)
imgspace_zf = np.fft.ifftn(kspace_cs, axes=(1,2,3))
imgspace_zf = np.fft.fftshift(imgspace_zf, axes=(1,2,3))
imgspace_zf = np.sum(imgspace_zf, axis=0)

"""
PLOTTING
"""
print(kspace_cs.shape)
plt.subplot(SUBPLOT_ROW,SUBPLOT_ROW,1)
plt.imshow(np.absolute(imgspace[:,:,SLICE,0]))
plt.subplot(SUBPLOT_ROW,SUBPLOT_ROW,2)
plt.imshow(np.absolute(imgspace_zf[:,:,SLICE,0]))
plt.subplot(SUBPLOT_ROW,SUBPLOT_ROW,3)
plt.imshow(np.absolute(kspace[0,:,:,SLICE,0]))
plt.subplot(SUBPLOT_ROW,SUBPLOT_ROW,4)
plt.imshow(np.absolute(kspace_cs[0,:,:,SLICE,0]))
plt.show()


