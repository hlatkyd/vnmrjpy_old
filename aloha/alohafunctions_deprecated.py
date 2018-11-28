#!/usr/local/bin/python3.6

import gc
import sys
import numpy as np
import cupy as cp
from scipy.linalg import hankel as scipyHankel
import matplotlib.pyplot as plt

"""
Collection of ALOHA helper functions.
More descriptions in aloha.py
"""
#numpy datatype
DTYPE = 'complex64'

def kx_ky_TV_weights(shape, rp, p):
    """
    Make kspace weights for ALOHA reconstruction
    in TV sparsity weights are j*w

    input:
        shape : numpy array, kspace shape
        rp : dictionary of recon parameters
        p : dictionary of procpar paramateres

    output:
        finished kspace weights in numpy array
    """

    cs_dim = rp['cs_dim']
    ro_dim = rp['ro_dim']
    weights = np.zeros(shape, dtype=DTYPE)
    kx_weights = [1j*(2*np.pi*i/shape[cs_dim[0]] - np.pi) \
                    for i in range(0,shape[cs_dim[0]])]
    ky_weights = [1j*(2*np.pi*i/shape[cs_dim[1]] - np.pi) \
                    for i in range(0,shape[cs_dim[1]])]
    weights_2d = np.outer(np.asarray(kx_weights), np.asarray(ky_weights))
    
    #plt.imshow(np.absolute(weights_2d), cmap='gray')
    #plt.show()

    for i in range(weights.shape[ro_dim]):
        for j in range(weights.shape[0]):
            weights[j,:,:,i,0] = weights_2d
    return weights


def weightedkspace2hankel(kspace, rp):
    """
    Makes hankel matrix from kspace data.
    
    input: 
        kspace : numpy arra of zerofilled kspace
        rp  :  dictionary of recon parameters
    """
    #---------------------ANGIO HELPERS ---------------------------------------
    def make_hankel_1slc_1rcvr(kslice, p,q,m,n):
        """
        Makes Hankel matrix out of 1 slice of k-space from 1 receiver
        Cs in both dimensions are supposed 
        k-space_slice is 2dim
        """
        # making inner Hankel
        inner_hankel = []  # is a list
        for j in range(kslice.shape[1]):
            cols = []
            for i in range(0,p):
                cols.append([kslice[k,j] for k in range(i,i+m-p+1)])

            inner_hankel.append(np.array(np.hstack(cols),dtype=DTYPE))
        # making outer Hankel

        outer_cols = []
        for i in range(0,q):
            outer_cols.append([inner_hankel[j] for j in range(i,i+n-q+1)])

        hankel = np.hstack(outer_cols)
        #print('hankel dtype : {}'.format(hankel.dtype))
        #print('inner hankel {}'.format(inner_hankel[0].shape))

        return hankel

    def make_hankel_allslc_1rcvr(kslab3d, p,q,m,n):
        """
        same as before but dont cut to slices
        readout axis=2
        """
        # making inner Hankel
        inner_hankel = []  # is a list
        for j in range(kslab3d.shape[1]):
            cols = []
            for i in range(0,p):
                cols.append(np.stack( \
                    [kslab3d[k,j,:] for k in range(i,i+m-p+1)], \
                        axis = 0))

            inner_hankel.append(np.array(np.stack(cols,axis=1)))

        # making outer Hankel

        outer_cols = []
        for i in range(0,q):
            outer_cols.append(np.concatenate(\
                        [inner_hankel[j] for j in range(i,i+n-q+1)],\
                        axis = 0))

        hankel = np.concatenate(outer_cols, axis=1)
        print('hankel 1 rcvr shap {}'.format(hankel.shape))
        return np.array(hankel,dtype=DTYPE)

    #--------------------------------------------------------------------------
    print('init kspace shape: {}'.format(kspace.shape))
    filter_size = rp['filter_size']
    p = filter_size[0]
    q = filter_size[1]
    m = kspace.shape[rp['cs_dim'][0]]
    n = kspace.shape[rp['cs_dim'][1]]
    
    # init full space low rank hankel

    if rp['recontype'] == 'kx-ky_angio':

        for rcvr in range(0,rp['rcvrs']):

            #kslice = kspace[rcvr,:,:,slc]
            kslab3d = kspace[rcvr,:,:,:,0]
            #full_slice = make_hankel_1slc_1rcvr(kslice, p,q,m,n)
                
            hankel_1rcvr = make_hankel_allslc_1rcvr(kslab3d, p,q,m,n)

            if rcvr == 0:
                hankel = hankel_1rcvr
            elif rcvr != 0:
                hankel = np.concatenate([hankel, hankel_1rcvr], axis=1)

            print('hankel1rcvr {}'.format(hankel_1rcvr.nbytes))
            print('hankel1rcvr dtype {}'.format(hankel_1rcvr.dtype))
            print('hankel {}'.format(hankel.nbytes))

            del kslab3d, hankel_1rcvr
            gc.collect()

        return hankel


        print('full slice shape {}'.format(full_slice.shape))

def make_solver_mask(hankel):

    mask = np.zeros(hankel.shape, dtype=bool)
    mask[hankel !=0] = 1
    return mask

def hankel2weightedkspace(hankel, rp):

    if rp['recontype'] == 'kx-ky_angio':
        pass
    pass

def unweigh_kspace(kspace_weighted, rp):

    pass

def NaN_fill(matrix):

    matrix[matrix == 0] = np.nan

    return matrix

# DEPRECATED CODE, MIGHT BE OF SOME USE THOUGH

    """
    def make_inner_hankel_1ch_1slc(kspace, m, p, rcvr, slc):
        #makes hankel matrix from one dimension of kspace data
        return: list of inner hankel matrices
        hankel_inner = []
        for j in range(kspace.shape[rp['cs_dim'][1]]):
            first_col = [kspace[rcvr,i,j,slc] for i in range(0,m-p+1)]
            last_row = [kspace[rcvr,i,j,slc] for i in range(m-p,m)]
            hankel_inner.append(scipyHankel(first_col, last_row)) 
        
        return hankel_inner

    def make_outer_hankel_1ch_1slc(hankel_inner, n, q, rcvr, slc):
        #Makes outer hankel matrix (in case of 2nd dim sparsity)
        first_col = [hankel_inner[j] for j in range(0,n-q+1)]
        last_row = [hankel_inner[j] for j in range(n-q,n)]
        
        return scipyHankel(first_col, last_row)
    """
