#!/usr/local/bin/python3.6

import numpy as cp
#import cupy as cp

"""
Functions for handling Hankel matrices in various ALOHA implementations

"""

DTYPE = 'complex64'

def make_pyramidal_weights_kt(slice2d_shape,rp):

    def haar_weights(s,w):
        return 1/cp.sqrt(2**s)*(-1j*2**s*w)/2*\
                (cp.sin(2**s*w/4)/(2**s*w/4))**2*cp.exp(-1j*2**s*w/2)
        
    weights = [] 

    for s in range(STAGES):
        
        w_samples = [k/2**s for k in range(0,slice2d.shape[0])]
        w_arr = np.array([haar_weights(s,i) for i in w_samples])
        weights.append(w_arr)

def compose_hankel_2d(slice3d,rp):
    """
    Make Hankel matrix, put onto GPU
    INPUT: slice2d_all_rcvrs : numpy.array[receivers, slice]
    OUTPUT: hankel : numpy.array (m-p)*(n-q) x p*q*rcvrs
    """
    # init on GPU
    slice3d = cp.array(slice3d, dtype=DTYPE)
    # annihilating filter size
    p = rp['filter_size'][0]
    q = rp['filter_size'][1]
    # hankel m n
    m = slice3d.shape[1]
    n = slice3d.shape[2]
    
    for rcvr in range(rp['rcvrs']):

        slice2d = slice3d[rcvr,...]

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

def hankel_completion_svd(hankel):

    svd = cp.linalg.svd(hankel)

    return svd

def decompose_hankel(hankel, rp):

    pass
