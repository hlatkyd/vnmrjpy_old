#!/usr/local/bin/python3.6

import sys
import os
sys.path.append('/home/david/dev/vnmrjpy')
import glob
import nibabel as nib
from writenifti import niftiWriter
import numpy as np
import matplotlib.pyplot as plt
import cshelper as csh
import pywt
import cvxopt

TESTDIR = '/home/david/dev/vnmrjpy/dataset/cs/ge3d_angio_HD_s_2018072604_HD_01.cs'
SLICE = 100
SUBPLOT_ROW = 2
SUBPLOT_COL = 2
#sparse axis = 1, 2

def load_data():

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
    kspace = np.vectorize(complex)(data_real, data_imag)
    """
    #TODO DC removal????
    """

    kspace_cs = np.multiply(kspace, mask)
    return (kspace, kspace_cs)

def make_imgspace_data(kspace, kspace_cs):

    imgspace = np.fft.ifftn(kspace, axes=(1,2,3))
    imgspace = np.fft.fftshift(imgspace, axes=(1,2,3))
    imgspace = np.sum(imgspace, axis=0)
    imgspace_zf = np.fft.ifftn(kspace_cs, axes=(1,2,3))
    imgspace_zf = np.fft.fftshift(imgspace_zf, axes=(1,2,3))
    imgspace_zf = np.sum(imgspace_zf, axis=0)

    return (imgspace, imgspace_zf)

def save_6d_kspace(kspace):
    """
    Utility to save 6d kspace into nifti for faster read/write
    ndarray([x,y,z,t,rcvrs,comp/real])
    """

    nib.

class CompressedSensingRecon():

    def __init__(self, data):
    
        pass

if __name__ == '__main__':

    (kspace, kspace_cs) = load_data()
    (imgspace, imgspace_zf) = make_imgspace_data(kspace, kspace_cs)



"""
PLOTTING
"""

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

"""
