#!/usr/local/bin/python3.6

import numpy as np
import cupy as cp

"""
Functions for handling Hankel matrices in various ALOHA implementations

"""

DTYPE = 'complex64'

def compose_hankel(slice2d_all_rcvrs,rp):
    """
    Make Hankel matrix, put onto GPU
    INPUT: slice2d_all_rcvrs : numpy.array[receivers, slice]
    OUTPUT: hankel : numpy.array (m-p)*(n-q) x p*q*rcvrs
    """
    # init on GPU
    slice2d_all_rcvrs = cp.array(slice2d_all_rcvrs, dtype=DTYPE)
    # annihilating filter size
    p = rp['filter_size'][0]
    q = rp['filter_size'][1]
    # hankel m n
    m = slice2d_all_rcvrs.shape[1]
    n = slice2d_all_rcvrs.shape[2]
    

    if rp['recontype'] == 'kx-ky_angio':

        for rcvr in range(rp['rcvrs']):

            slice2d = slice2d_all_rcvrs[rcvr,...]

            #make inner hankel list
            for j in range(0,n):
                # iterate in outer hankel elements
                for k in range(0,p):
                    # iterate inner hankel columns
                    col = cp.expand_dims(slice2d[k:m-p+k+1,j],axis=1)
                    if k == 0:
                        cols = col
                    else:
                        cols = cp.concatenate([cols,col],axis=1)
                if j == 0:
                    hankel = cp.expand_dims(cols,axis=0)
                else:
                    hankel = cp.concatenate(\
                        [hankel, cp.expand_dims(cols,axis=0)], axis=0)

            # make outer hankel
            for i in range(q):
                #col = cp.hstack([hankel[i:n-q+i,:,:]])
                col = cp.vstack([hankel[k,:,:] for k in range(i,n-q+i+1)])
                if i == 0:
                    cols = col
                else:
                    cols = cp.concatenate([cols,col], axis=1)
            # concatenating along the receivers
            if rcvr == 0:
                hankel_full = cols
            else:
                hankel_full = cp.concatenate([hankel_full, cols], axis=1)

        return hankel_full
        
    elif rp['recontype'] == 'k-t':
        pass
    elif rp['recontype'] == 'kx-ky':
        pass
    else:
        pass

def hankel_completion_svd(hankel):

    for i in range(50):

        svd = cp.linalg.svd(hankel)

def decompose_hankel(hankel, rp):

    pass
