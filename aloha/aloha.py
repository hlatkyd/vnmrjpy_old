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

#from matrix_completion import nuclear_norm_solve

from readfid import fidReader
from kmake import kSpaceMaker
from readprocpar import procparReader
from writenifti import niftiWriter

from hankelutils import *

# TESTING PARAMETERS

TESTDIR_ROOT = '/home/david/dev/vnmrjpy/dataset/cs/'
#TESTDIR = TESTDIR_ROOT + 'ge3d_angio_HD_s_2018072604_HD_01.cs'
#TESTDIR = TESTDIR_ROOT + 'gems_s_2018111301_axial_0_0_0_01.cs'
TESTDIR = TESTDIR_ROOT+'mems_s_2018111301_axial_0_0_0_01.cs'
PROCPAR = TESTDIR+'/procpar'
# undersampling dimension
CS_DIM = (1,4)  # phase and slice
RO_DIM = 2


FILTER_SIZE = (7,7)

"""
NOTE: testing angio data: rcvrs, phase1, phase2, read

NOTE gems data: 
"""

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
            elif 'mems' in self.p['pslabel']:
                recontype = 'k-t'
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
                    'recontype' : recontype,\
                    'timedim' : 4}
        print(self.rp)
        self.kspace_cs = np.array(kspace_cs, dtype='complex64')

    def recon(self):

        if self.rp['recontype'] == 'kx-ky_angio':
            for slc in range(self.kspace_cs.shape[self.rp['ro_dim']]):
                slice2d_all_rcvrs = self.kspace_cs[:,:,:,slc,0]
                hankel = compose_hankel(slice2d_all_rcvrs, self.rp)
                svd = hankel_completion_svd(hankel)

        elif self.rp['recontype'] == 'k-t':

            STAGES = 3

            weights = make_pyramidal_weights_kt(slice2d_shape, rp, stages)

            def pyramidal_solve(slice3d):
    
                for stage in range(STAGES)
                
                    #TODO
                    # init from previous stage
                    #kspace_weighing     
                    #hankel formation
                    #rank estimation
                    #Hankel completion (ADMM)
                # return
                #kspace unweighing (average across all)

                pass

    
            for slc in range(kspace_cs.shape[3]):
                for x in range(kspace_cs.shape[rp['cs_dim'][0]])
                    slice3d = kspace[:,:,x,slc,:]
                    slice3d_completed = pyramidal_solve(slice3d)
                    fin.append(slice3d_completed)
                

        elif self.rp['recontype'] == 'kx-ky':

            pass

#----------------------------------FOR TESTING---------------------------------

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
    #plt.imshow(np.real(kspace_cs[0,:,:,10,4]), cmap='gray')
    #plt.show()
    return kspace_cs

if __name__ == '__main__':

    kspace_cs = load_test_data()

    aloha = ALOHA(PROCPAR, kspace_cs)

    aloha.recon()

