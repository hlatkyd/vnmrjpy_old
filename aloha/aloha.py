#!/usr/local/bin/python3.6

import sys
import numpy as np
import cupy as cp
import glob
import nibabel as nib
import matplotlib.pyplot as plt

sys.path.append('/home/david/dev')
sys.path.append('/home/david/dev/vnmrjpy')
sys.path.append('/home/david/dev/vnmrjpy/aloha')

from readfid import fidReader
from kmake import kSpaceMaker
from readprocpar import procparReader
from writenifti import niftiWriter

from alohafunctions import *
from lowranksolvers import *

# TESTING PARAMETERS

TESTDIR = '/home/david/dev/vnmrjpy/dataset/cs/ge3d_angio_HD_s_2018072604_HD_01.cs'
PROCPAR = '/home/david/dev/vnmrjpy/dataset/cs/ge3d_angio_HD_s_2018072604_HD_01.fid/procpar'
# undersampling dimension
CS_DIM = (1,2)
RO_DIM = 3
FILTER_SIZE = (7,7)

"""
NOTE:

testing angio data: rcvrs, phase1, phase2, read

"""

def load_test_data():

    imag = []
    real = []

    mask_img = nib.load(TESTDIR+'/kspace_mask.nii.gz')
    afiine = mask_img.affine
    mask = mask_img.get_fdata()

    for item in sorted(glob.glob(TESTDIR+'/kspace_imag*')):
        data = nib.load(item).get_fdata()
        data = np.multiply(data, mask)
        imag.append(data)

    for item in sorted(glob.glob(TESTDIR+'/kspace_real*')):
        data = nib.load(item).get_fdata()
        data = np.multiply(data, mask)
        real.append(data)

    imag = np.asarray(imag)
    real = np.asarray(real)
    kspace_cs = np.vectorize(complex)(real, imag)

    print('kspace shape : {}'.format(kspace_cs.shape))
    print('mask shape : {}'.format(mask.shape))

    #plt.imshow(np.real(kspace_cs[0,:,:,100,0]), cmap='gray')
    #plt.show()

    return kspace_cs

class ALOHA():
    """
    Class for compressed sensing completion using ALOHA:

    ref: Jin et al.: A general framework for compresed sensing and parallel
        MRI using annihilation filter based low-rank matrix completion (2016)
    
    Process:

        1. kspace weighing
        2. pyramidal decomposition in case of wavelet transform
        3. Hankel matrix formation, RANK ESTIMATION (MAYBE)
        4. matrix completion (Multiple approaches maybe)
        5. kspace unweighing

    """

    def __init__(self, procpar, kspace_cs, reconpar=None):
        """
        INPUT:
            procpar : path to procpar file
            
            kspace_cs : zerofilled cs kspace in numpy array

            reconpar: dictionary, ALOHA recon parameters
                    keys:
                        filter_size
                        rcvrs
                        cs_dim
                        recontype

        """
        def get_recontype(reconpar):

            if 'angio' in self.p['pslabel']:

                recontype = 'kx-ky_angio'

            return recontype

        def get_reconpar():

            pass

        self.p = procparReader(procpar).read() 
        recontype = get_recontype(reconpar)
        rcvrs = self.p['rcvrs'].count('y')

        self.rp = {'filter_size' : FILTER_SIZE ,\
                    'cs_dim' : CS_DIM ,\
                    'ro_dim' : RO_DIM, \
                    'rcvrs' : rcvrs , \
                    'recontype' : recontype }
        
        print(self.rp)
        self.kspace_cs = kspace_cs

    def recon(self):

        if self.rp['recontype'] == 'kx-ky_angio':

            weights = kx_ky_TV_weights(self.kspace_cs.shape, self.rp, self.p)
    
            kspace_cs_weighted = np.multiply(kspace_cs, weights)

            hankel_low_rank = weightedkspace2hankel(kspace_cs_weighted, self.rp)

            
        elif self.rp['recontype'] == 'k-t':

            pass

        elif self.rp['recontype'] == 'kx-ky':

            pass



if __name__ == '__main__':

    kspace_cs = load_test_data()

    aloha = ALOHA(PROCPAR, kspace_cs)

    aloha.recon()

