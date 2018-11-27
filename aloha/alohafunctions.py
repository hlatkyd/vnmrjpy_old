#!/usr/local/bin/python3.6

import gc
import sys
import numpy as np
import cupy as cp
from scipy.linalg import hankel
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

    #----------------------------HANKEL MAKING FUNC----------------------------

        def make_hankel_outer_1ch(kspace, p,q,m,n, rcvr):

            slab3d = np.array(kspace[rcvr,:,:,:],dtype=DTYPE)
            
            # making inner hankel

            hankel_inner = []

            for j in range(0, slab3d.shape[rp['ro_dim']-1]):

                hankel_inner_cols = []

                for i in range(0, p):

                    # making 1 column in hankel
                    hankel_inner_1col_list = [slab3d[k,j,:] for k in \
                                range(i,i+m-p+1)]

                    hankel_inner_cols.append(np.array(\
                            np.vstack(hankel_inner_1col_list),dtype=DTYPE))
                hankel_inner.append(np.array(\
                            np.stack(hankel_inner_cols, axis=1),dtype=DTYPE))
            print('inner hankel cols[0] shape :{}'.format(hankel_inner_cols[0].shape))
            print('inner hankel cols :{}'.format(len(hankel_inner_cols)))
            print('inner hankel[0] shape full :{}'.format(hankel_inner[0].shape))
            print('inner hankel full length :{}'.format(len(hankel_inner)))
            #hankel_inner.append(np.concatenate(hankel_inner_col, axis=0))

            # deleting garbage
    
            del hankel_inner_1col_list
            del hankel_inner_cols

            # making outer hankel

            hankel_outer_cols = []
        
            for i in range(0, q):
                hcol_list = [np.array(hankel_inner[j],dtype=DTYPE)\
                                 for j in range(i, i+n-q+1)]
                hankel_outer_1col = np.concatenate(hcol_list, axis=0)
                hankel_outer_cols.append(np.array(hankel_outer_1col, dtype=DTYPE)) 

            hankel_outer_1ch = np.array(np.concatenate(hankel_outer_cols,\
                                                        axis=1),\
                                                        dtype=DTYPE)
            print('outer: {}'.format(hankel_outer_1ch.shape))
            print('size : {}'.format(hankel_outer_1ch.nbytes))
            # deleting garbage

            return hankel_outer_1ch

        #----------------------------------------------------------------------

        if kspace.shape[-1] == 1:

            kspace = kspace[...,0]

        else:
            print('weighted2hankel: Could not remove last dimension')

        hankel_low_rank = []

        for rcvr in range(kspace.shape[0]):

            hankel_low_rank.append(make_hankel_outer_1ch(kspace, p,q,m,n,rcvr))
        
        hankel_low_rank = np.concatenate(hankel_low_rank,axis=1)

        return hankel_low_rank
        """
            slab3d = np.array(kspace[rcvr,:,:,:],dtype=DTYPE)
            
            # making inner hankel

            hankel_inner = []

            for j in range(0, slab3d.shape[rp['ro_dim']-1]):

                hankel_inner_cols = []

                for i in range(0, p):

                    # making 1 column in hankel
                    hankel_inner_1col_list = [slab3d[k,j,:] for k in \
                                range(i,i+m-p+1)]

                    hankel_inner_cols.append(np.array(\
                            np.vstack(hankel_inner_1col_list),dtype=DTYPE))
                hankel_inner.append(np.array(\
                            np.stack(hankel_inner_cols, axis=1),dtype=DTYPE))
            print('inner hankel cols[0] shape :{}'.format(hankel_inner_cols[0].shape))
            print('inner hankel cols :{}'.format(len(hankel_inner_cols)))
            print('inner hankel[0] shape full :{}'.format(hankel_inner[0].shape))
            print('inner hankel full length :{}'.format(len(hankel_inner)))
            #hankel_inner.append(np.concatenate(hankel_inner_col, axis=0))

            # deleting garbage
    
            del hankel_inner_1col_list
            del hankel_inner_cols

            # making outer hankel

            hankel_outer_cols = []
        
            for i in range(0, q):
                hcol_list = [np.array(hankel_inner[j],dtype=DTYPE)\
                                 for j in range(i, i+n-q+1)]
                hankel_outer_1col = np.concatenate(hcol_list, axis=0)
                hankel_outer_cols.append(np.array(hankel_outer_1col, dtype=DTYPE)) 

            hankel_outer.append(np.array(np.concatenate(hankel_outer_cols,\
                                                        axis=1),\
                                                        dtype=DTYPE))
            print('outer: {}'.format(hankel_outer[0].shape))

            print('size : {}'.format(hankel_outer[0].nbytes))
            # deleting garbage

            del hankel_inner
            del hankel_outer_cols

            gc.collect()

        hankel_low_rank = np.concatenate(hankel_outer,axis=1)
        #print('full hanekl shape: {}'.format(hankel_full.shape))
            #print('inner hankel shape :{}'.format(hankel_inner[0].shape))
        """

def hankel2weightedkspace(hankel, rp):

    pass

def unweigh_kspace(kspace_weighted, rp):

    pass
