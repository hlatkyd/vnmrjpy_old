#!/usr/local/bin/python3

import copy
import numpy as np
import cupy as cp
from hankelutils import *

DTYPE = 'complex64'

class ADMM():
    """
    Class for problem solving with Alternating Direction Method of Multipliers
    Used for a low rank matrix completion step after, U,V in H = U*V' have been
    initialized by LMaFit. Used in ALOHA MRI reconstruction framework.
    -------------------------------------------------------------------------
    Input:

    -------------------------------------------------------------------------
    Methods:
        
        solve
        solve_CUDA
    """

    def __init__(self,U,V,hankel_cs,slice3d_cs,stage,rp):

        U = np.matrix(U)
        V = np.matrix(V)
        slice3d_shape = slice3d_cs.shape
        hankel_cs = np.matrix(hankel_cs)
        hankel_pre = np.matrix(U.dot(V.H))

        self.hankel_mask = copy.deepcopy(hankel_cs)
        self.hankel_mask[hankel_cs != 0] = 1
        self.hankel_mask_inv = np.ones(self.hankel_mask.shape) - self.hankel_mask
        self.initpars = (U,V,hankel_cs,hankel_pre,slice3d_orig,slice3d_shape,\
                        stage,rp)
        self.decomp_factors = make_hankel_decompose_factors(slice3d_shape,rp)

    def solve(self,mu=1000,noiseless=True,max_iter=100):

        (U,V,hankel_cs,hankel_pre,slice3d_shape,slice3d_orig,s,rp) = self.initpars

        slice3d_orig_part = slice3d_orig 
        # init output hankel
        hankel = hankel_pre
        # init lagrangian update
        lagr = np.zeros(hankel_pre.shape,dtype=DTYPE)

        if self.rp['recontype'] == 'k-t':
    

            for _ in range(max_iter):

                hankel_inferred_part = np.multiply(U.dot(V.H)-lagr,\
                                                    self.hankel_mask_inv)  
                slice3d_inferred_part = decompose_hankel_2d(hankel_inferred_part,\
                                                    slice3d_shape,s,factors,rp)
                slice3d = slice3d_orig_part + slice3d_inferred_part
                hankel = compose_hankel_2d(slice3d,rp)

                U = mu*(hankel+lagr).dot(V).dot(np.linalg.inv\
                                        (np.eye(us)+mu*V.H.dot(V)))
                V = mu*(hankel+lagr).dot(U).dot(np.linalg.inv\
                                        (np.eye(vs)+mu*U.H.dot(U)))

                lagr = hankel - U.dot(V.H) + lagr

            return U.dot(V.H)

        elif self.rp['recontype'] == 'kx-ky':
        
            pass

        elif self.rp['recontype'] == 'kx-ky_angio':
        
            pass
    
    def solve_CUDA(self,mu=1000,noiseless=True,max_iter=100):

        pass
