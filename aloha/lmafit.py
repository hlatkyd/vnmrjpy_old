#!/usr/local/bin/python3.6

import imageio
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr

PICDIR = '/home/david/dev/vnmrjpy/testpics'

class LMaFit():

    def __init__(self,zfdata,fitpars=None):

        (m,n) = zfdata.shape
        if fitpars == None:
            k = 5
            tol = 1.25e-5
            maxit = 500
            rank_strategy = 'increase'

        datanrm = np.max([1,np.linalg.norm(zfdata,'fro')])
        objv = np.zeros(maxit)
        RR = np.ones(maxit)
        # init
        Z = zfdata
        X = np.zeros((m,k))
        Y = np.eye(k,n)
        Res = zfdata
        res = datanrm
        reschg_tol = 0.5*tol
        # parameters for alf
        alf = 0
        increment = 0.1
        #rank estimation parameters
        itr_rank = 0
        minitr_reduce_rank = 5
        maxitr_reduce_rank = 50
        tau_limit = 10
        rank_incr = 3
        rank_max = 50

        datamask = copy.deepcopy(zfdata)
        datamask[datamask != 0] = 1

        self.initpars = (zfdata,m,n,k,tol,maxit,rank_strategy,datanrm,objv,\
                        RR,Z,X,Y,Res,res,reschg_tol,alf,increment,itr_rank,\
                        minitr_reduce_rank,maxitr_reduce_rank,tau_limit,\
                        datamask,rank_incr,rank_max)

    def solve_mc(self):

        def rank_check(R,reschg,tol):
            
            diag = np.diag(R)
            d_hat = [diag[i]/diag[i+1] for i in range(len(diag)-1)]
            tau = (len(diag)-1)*max(d_hat)/(sum(d_hat)-max(d_hat))

            if reschg < 10*tol:
                ind_string = 'increase'
            else:
                ind_string = 'stay'
            return ind_string

        def increase_rank(X,Y,Z,rank_incr,rank_max):
            
            k = X.shape[1]
            k_new = min(k+rank_incr,rank_max)

            m = X.shape[0]
            n = Y.shape[1]
            X_new = np.zeros((m,k_new))
            Y_new = np.eye(k_new,n)            
            X_new[:,:k] = X
            Y_new[:k,:] = Y
            Z_new = X.dot(Y)
            print('new rank {}'.format(k_new))
            return X_new, Y_new, Z_new

        # -------------------INIT------------------------

        (zfdata,m,n,k,tol,maxit,rank_strategy,datanrm,objv,\
        RR,Z,X,Y,Res,res,reschg_tol,alf,increment,itr_rank,\
        minitr_reduce_rank,maxitr_reduce_rank,tau_limit,\
                    datamask, rank_incr,rank_max) = self.initpars

        # --------------MAIN ITERATION--------------------

        for iter_ in range(maxit):
            itr_rank += 1

            X0 = copy.deepcopy(X)
            Y0 = copy.deepcopy(Y)
            Res0 = copy.deepcopy(Res)
            res0 = copy.deepcopy(res)
            Z0 = copy.deepcopy(Z)
            X = Z.dot(Y.T)
            X_to_qr = copy.deepcopy(X)
            X, R, P = qr(X_to_qr,pivoting=True,mode='economic')
            Y = X.T.dot(Z)
            Z = X.dot(Y)
            Res = np.multiply(zfdata-Z,datamask)
            res = np.linalg.norm(Res,'fro')
            relres = res / datanrm
            ratio = res / res0
            reschg = np.abs(1-res/res0)
            RR[iter_] = ratio
            # adjust alf
            if ratio >= 1.0:
                increment = np.max([0.1*alf,0.1*increment])
                X = copy.deepcopy(X0)
                Y = copy.deepcopy(Y0)
                Res = copy.deepcopy(Res0)
                res = copy.deepcopy(res0)
                relres = res / datanrm
                alf = 0
                Z = copy.deepcopy(Z0)
            elif ratio > 0.7:
                increment = max(increment,0.25*alf)
                alf = alf + increment 
            objv[iter_] = relres
            # check stopping
            if ((reschg < reschg_tol) and ((itr_rank > minitr_reduce_rank) \
                                    or (relres < tol))):
                print('Stopping crit achieved')
                break

            # rank adjustment
            rankadjust = rank_check(R,reschg,tol)
            if rankadjust == 'increase':
                X,Y,Z = increase_rank(X,Y,Z,rank_incr,rank_max)

            Zknown = zfdata + alf*Res
            Z = Z - np.multiply(Z,datamask) + Zknown

        obj = objv[:iter_]

        return X, Y, [obj, RR, iter_, relres, reschg] 
            

# -----------------------TESTING----------------------------------------------

def plot_test_data(images2d):

    n = len(images2d)
    for num, img in enumerate(images2d):
        plt.subplot(1,n,num+1)
        plt.imshow(img,cmap='gray',vmin=0,vmax=256)
    plt.show()

def make_test_data():
    #a = np.array([np.sin(i/3) for i in range(100)])
    #b = np.array([np.sin(i/3) for i in range(100)])
    a = np.array([i for i in range(100)])    
    b = np.array([i/3 for i in range(100)])    
    A = np.outer(a,b)
    mask = np.random.rand(A.shape[0],A.shape[1])
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    A_masked = np.multiply(A,mask)    

    return A, A_masked, mask

def load_boat():

    im = imageio.imread(PICDIR+'/boat.png')
    mask = np.random.rand(im.shape[0],im.shape[1])
    mask[mask >= 0.7] = 1
    mask[mask < 0.7] = 0
    im_masked = np.multiply(im,mask)    

    return im, im_masked, mask

def lmafit_input_preproc(A_masked):
    """Preprocess masked data for LMaFit"""
    (m,n) = A_masked.shape
    vec1d = np.reshape(A_masked,(A_masked.size),order='f')
    known = np.nonzero(vec1d)[0]
    data = vec1d[known]
    return m, n, known, data

if __name__ == '__main__':

    im, im_masked, mask = load_boat()
    slv = LMaFit(im_masked)
    X, Y, out = slv.solve_mc()
    plot_test_data([im, im_masked, mask, X.dot(Y)])


