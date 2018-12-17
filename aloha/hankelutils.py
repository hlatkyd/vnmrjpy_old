#!/usr/local/bin/python3.6

import numpy as np
#import cupy as cp
import matplotlib.pyplot as plt
import copy

"""
Functions for handling Hankel matrices in various ALOHA implementations

"""

DTYPE = 'complex64'
def checkutils(weights_list,factors):

    for num, item in enumerate(weights_list):
        plt.subplot(1,len(weights_list),num+1)
        plt.plot(np.absolute(item))
    plt.show()


def checkhankels(hankel_list):

    pass

def restore_center(slice3d, slice3d_orig):
    
    ind = slice3d_orig.shape[1]

    center = slice3d_orig[:,ind//2-1:ind//2+2]

    slice3d[:,ind//2-1:ind//2+2,:] = center

    return slice3d

def finalize_pyramidal_stage(comp_stage3d, comp3d, slice3d, s, rp):

    m = slice3d.shape[1]  # phase dim
    # replace kspace elements with the completed ones from the stage
    comp3d[:,m//2-m//2**(s+1):m//2+m//2**(s+1),:] = comp_stage3d
    return comp3d

def kspace_pyramidal_init(slice3d, s):
    """
    Initialize stage s reduced kspace from previous stage
    No reduction at stage 0
    Makes sure the freq=0 components are OK
    """
    kx_ind = slice3d.shape[1]
    if s == 0:
        kspace_init = slice3d
    elif s != 0:
        kspace_init = slice3d[:,kx_ind//2-kx_ind//2**(s+1): \
                                kx_ind//2+kx_ind//2**(s+1),:] 

    center = kspace_init[:,kx_ind//2-1:kx_ind//2,:]

    return kspace_init, center

def apply_pyramidal_weights_kxt(slice3d, weights_s, rp):

    weights_s = np.expand_dims(weights_s,axis=0)
    weights_s = np.expand_dims(weights_s,axis=-1)
    weights_s = np.repeat(weights_s,slice3d.shape[0],axis=0)
    weights_s = np.repeat(weights_s,slice3d.shape[2],axis=-1)

    return np.multiply(slice3d, weights_s,dtype=DTYPE)

def remove_pyramidal_weights_kxt(slice3d, center, weights_s):

    weights_s = np.expand_dims(weights_s,axis=0)
    weights_s = np.expand_dims(weights_s,axis=-1)
    weights_s = np.repeat(weights_s,slice3d.shape[0],axis=0)
    weights_s = np.repeat(weights_s,slice3d.shape[2],axis=-1)
    slice3d_unweighted = np.divide(slice3d, weights_s,dtype=DTYPE)

    #slice3d_unweighted[:,slice3d.shape[1]//2-1,:] = center
    return slice3d_unweighted

def make_pyramidal_weights_kxt(kx_len, t_len ,rp):
    """
    Makes weights for wavelet sparse image. The weights are from the 
    Fourier transform of Haar wavelets
    INPUT: kx_len : cs array initial length (kspace shape along phase
            encode direction)
            t_len : kspace shape along time direction
            rp :  reconparameters dictionary
    OUTPUT : weights_list: list of weight arrays at various pyramidal
                            decomp stages (s)
    """
    def haar_weights(s,w):
        if w!=0:
            we = 1/np.sqrt(2**s)*(-1j*2**s*w)/2*\
                (np.sin(2**s*w/4)/(2**s*w/4))**2*np.exp(-1j*2**s*w/2)
        else:
            we = 0.001  # just something small not to divide by 0 accidently
        return we
        
    weights_list = [] 

    for s in range(rp['stages']):
        
        w_samples = [2*np.pi/kx_len*k for k in\
                     range(-int(kx_len/2**(s+1)),int(kx_len/2**(s+1)))]
        w_arr = np.array([haar_weights(s,i) for i in w_samples],dtype=DTYPE)
        weights_list.append(w_arr)

    return weights_list

def compose_hankel_2d(slice3d,rp):
    """
    Make Hankel matrix, put onto GPU
    INPUT: slice2d_all_rcvrs : numpy.array[receivers, slice]
    OUTPUT: hankel : numpy.array (m-p)*(n-q) x p*q*rcvrs
    """
    slice3d = np.array(slice3d, dtype=DTYPE)
    # annihilating filter size
    p = rp['filter_size'][0]
    q = rp['filter_size'][1]
    # hankel m n
    m = slice3d.shape[1]
    n = slice3d.shape[2]
   
    if rp['virtualcoilboost'] == False:
        receiverdim = int(rp['rcvrs'])
    elif rp['virtualcoilboost'] == True:
        receiverdim = int(rp['rcvrs']*2)
 
    for rcvr in range(receiverdim):

        slice2d = slice3d[rcvr,...]

        #make inner hankel list
        for j in range(0,n):
            # iterate in outer hankel elements
            for k in range(0,p):
                # iterate inner hankel columns
                col = np.expand_dims(slice2d[k:m-p+k+1,j],axis=1)
                if k == 0:
                    cols = col
                else:
                    cols = np.concatenate([cols,col],axis=1)
            if j == 0:
                hankel = np.expand_dims(cols,axis=0)
            else:
                hankel = np.concatenate(\
                    [hankel, np.expand_dims(cols,axis=0)], axis=0)

        # make outer hankel
        for i in range(q):
            #col = cp.hstack([hankel[i:n-q+i,:,:]])
            col = np.vstack([hankel[k,:,:] for k in range(i,n-q+i+1)])
            if i == 0:
                cols = col
            else:
                cols = np.concatenate([cols,col], axis=1)
        # concatenating along the receivers
        if rcvr == 0:
            hankel_full = cols
        else:
            hankel_full = np.concatenate([hankel_full, cols], axis=1)

    return hankel_full

def decompose_hankel_2d(hankel,slice3d_shape,s, factors, rp):
    """
    Decompose reconstructed hankel matrix into original kspace.
    Kspace values are the according averaged hankel values
    INPUT : hankel: np.array (m-p)*(n-q) x p*q*rcvrs
    OUTPUT : np.array([receivers,dim1,dim2])
    """
    #orig dimensions
    n = slice3d_shape[2]
    m = slice3d_shape[1]//2**s
    # init return kspace slab
    slice3d_stage_s = np.zeros((rp['rcvrs'],m,n),dtype=DTYPE)
    p = rp['filter_size'][0]
    q = rp['filter_size'][1]
    #print('n , m, p, q : {}, {}, {}, {}'.format(n,m,p,q))
    #inner hankel dimension:
    ih_col = m-p+1
    ih_row = p
    (factor_inner, factor_outer) = factors
    # ------------decomposing outer hankel---------------------
    if rp['virtualcoilboost'] == False:
        receiverdim = int(rp['rcvrs'])
    elif rp['virtualcoilboost'] == True:
        receiverdim = int(rp['rcvrs']*2)

    for i in range(receiverdim):
        hankel_1rcvr = hankel[:,hankel.shape[1]//rp['rcvrs']*i:\
                            hankel.shape[1]//rp['rcvrs']*(1+i)]
        inner_hankel_arr = np.zeros((n,ih_col,ih_row),dtype=DTYPE)
        # how many occurrences of H_inner[j] are in outer hankel
        for j in range(0,q):
            # for each slab in outer hankel
            fill_arr = np.zeros((n,ih_col,ih_row),dtype=DTYPE) 
            cols = [hankel_1rcvr[(m-p+1)*k:(m-p+1)*(k+1),p*j:p*(j+1)] \
                    for k in range(0,n-q+1)]
            cols = np.array(cols)
            fill_arr[j:n-q+j+1,:,:] = cols
            inner_hankel_arr = inner_hankel_arr + fill_arr
        # division by the multiples var to get the averages
        inner_hankel_avg_arr = np.divide(inner_hankel_arr,factor_outer[s])
        # avg: array of inner hankels
        # ----------------decomposing inner hankels------------
        for j in range(0,n):
            inner_hank_j = inner_hankel_avg_arr[j,:,:]
            hankel_j_arr = np.zeros(m,dtype=DTYPE)
            cols = [inner_hank_j[:,k] for k in range(inner_hank_j.shape[1])]
            cols = np.array(cols, dtype=DTYPE)
            for k in range(0,len(cols)):
                fill_arr = np.zeros(m,dtype=DTYPE)
                fill_arr[k:m-p+k+1] = cols[k]
                hankel_j_arr = hankel_j_arr + fill_arr
            hankel_j_arr_avg = np.divide(hankel_j_arr,factor_inner[s])
            slice3d_stage_s[i,:,j] = hankel_j_arr_avg
    return slice3d_stage_s

def make_hankel_decompose_factors(slice3d_shape, rp):
    
    # make 'multiples in decompose hankel2d beforehand'
    stages = rp['stages']
    factor_outer = []
    factor_inner = []
    for s in range(stages):
        # ------------make outer factors------------
        m = slice3d_shape[1]//2**s
        p = rp['filter_size'][0]
        n = slice3d_shape[2]
        q = rp['filter_size'][1]
        #inner hankel dimension:
        ih_col = m-p+1
        ih_row = p
        # making
        multiples = np.zeros(n,dtype=int)
        multiples[n-q:n] = np.flip(np.array([i for i in range(1,q+1)]),axis=0)
        multiples[0:q-1] = np.array([i for i in range(1,q)])
        multiples[q:n-q] = np.array([q for i in range(q,n-q+1)])
        multiples = np.expand_dims(multiples,1)
        multiples = np.expand_dims(multiples,2)
        multiples = np.repeat(multiples,ih_col,axis=1)
        multiples = np.repeat(multiples,ih_row,axis=2)
        factor_outer.append(multiples)
        # ------------make inner factors ------------
        multiples = np.zeros(m,dtype=int)
        multiples[m-p:m] = np.flip(np.array([i for i in range(1,p+1)]),axis=0)
        multiples[0:p-1] = np.array([i for i in range(1,p)])
        multiples[p-1:m-p] = np.array([p for i in range(p,m-p+1)])
        factor_inner.append(multiples)
    return (factor_inner, factor_outer)

def combine_stages():

    pass

def remove_weights_kxt():

    pass

