#!/usr/local/bin/python3.6

import numpy as np
import cupy as cp
from scipy.linalg import hankel
import matplotlib.pyplot as plt

"""
Collection of ALOHA helper functions.
More descriptions in aloha.py
"""

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
    weights = np.zeros(shape, dtype=complex)
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

    print('init kspace shape: {}'.format(kspace.shape))
    filter_size = rp['filter_size']
    p = filter_size[0]
    q = filter_size[1]
    m = kspace.shape[rp['cs_dim'][0]]
    n = kspace.shape[rp['cs_dim'][1]]

    if rp['recontype'] == 'kx-ky_angio':

        if kspace.shape[-1] == 1:

            kspace = kspace[...,0]

        else:
            print('weighted2hankel: Could not remove last dimension')

        hankel_shape = ((n-q+1)*(m-p+1),q*p)

        for rcvr in range(kspace.shape[0]):
            
            hankel_1ch_cols = []
            hankel_1ch = []
            for i in range(kspace.shape[rp['ro_dim']]):

                slice2d = kspace[rcvr,:,:,i]
                hankel_j = []
                for j in range(slice2d.shape[1]):
                
                    first_col = slice2d[:m-p,j]
                    last_row = slice2d[m-p:m-1,j]
                    hankel_j.append(hankel(first_col, last_row))

            # making hankel_1ch 1 col-by-col
            for k in range(0, q+1):    
                hankel_1ch_cols.append(np.vstack([hankel_j[j] \
                                        for j in range(k, k+n-q)]))
            print('hankel_1ch_col shape: {}'.format(hankel_1ch_cols[0].shape))
            hankel_1ch.append(np.hstack(hankel_1ch_cols))
        print('hanekl 1ch list len : {}'.format(len(hankel_1ch)))
        print('hankel_1ch shape: {}'.format(hankel_1ch[0].shape))
        # stacking 
        hankel_low_rank = np.hstack(hankel_1ch)
        print('final hankel shape: {}'.format(hankel_low_rank.shape))

        return hankel_low_rank

def hankel2weightedkspace(hankel, rp):

    pass

def unweigh_kspace(kspace_weighted, rp):

    pass
