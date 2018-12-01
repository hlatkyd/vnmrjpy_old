#!/usr/local/bin/python3.6

import sys
import numpy as np
import cupy as cp
import glob
import nibabel as nib
import matplotlib.pyplot as plt
import timeit
import time

sys.path.append('/home/david/dev')
sys.path.append('/home/david/dev/vnmrjpy')
sys.path.append('/home/david/dev/vnmrjpy/aloha')

#from matrix_completion import nuclear_norm_solve

from fancyimpute import SoftImpute
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
STAGES = 3
FILTER_SIZE = (7,5)

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
                    'timedim' : 4,\
                    'stages' : STAGES}
        print(self.rp)
        self.kspace_cs = np.array(kspace_cs, dtype='complex64')

    def recon(self):

        if self.rp['recontype'] == 'kx-ky_angio':
            for slc in range(self.kspace_cs.shape[self.rp['ro_dim']]):
                slice2d_all_rcvrs = self.kspace_cs[:,:,:,slc,0]
                hankel = compose_hankel(slice2d_all_rcvrs, self.rp)
                svd = hankel_completion_svd(hankel)

        elif self.rp['recontype'] == 'k-t':

            x_len = kspace_cs.shape[self.rp['cs_dim'][0]]
            t_len = kspace_cs.shape[self.rp['cs_dim'][1]]
            #each element of weight list is an array of weights in stage s
            weights_list = make_pyramidal_weights_kxt(x_len, t_len, self.rp)
            def pyramidal_solve(slice3d):
                """
                Solves a k-t slice: dim0=receivers,dim1=kx,dim2=t
                """ 
                kspace_complete_stage = slice3d
                for stage in range(self.rp['stages']):
                    #TODO
                    # init from previous stage
                    kspace_init = kspace_pyramidal_init(kspace_complete_stage,\
                                                        stage)
                    #kspace_weighing     
                    kspace_weighted = apply_pyramidal_weights_kxt(kspace_init,\
                                                        weights_list[stage],\
                                                        self.rp)
                    #hankel formation
                    hankel = compose_hankel_2d(kspace_weighted,self.rp)
                    decompose_hankel_2d(hankel,kspace_init.shape,self.rp)
                    return
                    #svd = cp.linalg.svd(cp.array(hankel))
                    #svd = np.linalg.svd(hankel)
                    #rank estimation
                    #Hankel completion (ADMM)
                    #kspace_complete_stage = complete_hankel()
                    #kspace_complete = decompose_hankel2d(hankel)
                    #kspace_complete = remove_weights_kxt()

                    # just for testrun .... 
                    kspace_complete_stage = kspace_init
                # return
                #kspace unweighing (average across all)

    
            for slc in range(self.kspace_cs.shape[3]):
                for x in range(self.kspace_cs.shape[self.rp['cs_dim'][0]]):
                    slice3d = self.kspace_cs[:,:,x,slc,:]
                    slice3d_completed = pyramidal_solve(slice3d)
                    #fin.append(slice3d_completed)
                

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
    start_time = time.time()
    aloha.recon()
    print('elapsed time {}'.format(time.time()-start_time))

