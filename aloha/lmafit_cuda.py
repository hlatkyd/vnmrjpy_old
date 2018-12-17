#!/usr/local/bin/python3.6

import imageio
import copy
import cupy as cp
import matplotlib.pyplot as plt
from scipy.linalg import qr
import time

PICDIR = '/home/david/dev/vnmrjpy/testpics'

class LMaFit():

    def __init__(self,zfdata,fitpars=None):

        (m,n) = zfdata.shape
        if fitpars == None:
            k = 2
            tol = 1.25e-5
            maxit = 500
            rank_strategy = 'increase'

        datanrm = cp.linalg.norm(zfdata,'fro')
        objv = cp.zeros(maxit)
        RR = cp.ones(maxit)
        # init
        Z = zfdata
        X = cp.zeros((m,k))
        Y = cp.eye(k,n)
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
            
            #diag = cp.diag(R)
            #d_hat = [diag[i]/diag[i+1] for i in range(len(diag)-1)]
            #tau = (len(diag)-1)*max(d_hat)/(sum(d_hat)-max(d_hat))

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
            X_new = cp.zeros((m,k_new))
            Y_new = cp.eye(k_new,n)            
            X_new[:,:k] = X
            Y_new[:k,:] = Y
            Z_new = X.dot(Y)
            return X_new, Y_new, Z_new

        # -------------------INIT------------------------

        (zfdata,m,n,k,tol,maxit,rank_strategy,datanrm,objv,\
        RR,Z,X,Y,Res,res,reschg_tol,alf,increment,itr_rank,\
        minitr_reduce_rank,maxitr_reduce_rank,tau_limit,\
                    datamask, rank_incr,rank_max) = self.initpars

        # --------------MAIN ITERATION--------------------

        for iter_ in range(maxit):
            itr_rank += 1

            X0 = cp.copy(X)
            Y0 = cp.copy(Y)
            Res0 = cp.copy(Res)
            res0 = cp.copy(res)
            Z0 = cp.copy(Z)
            X = Z.dot(Y.T)
            X, R = cp.linalg.qr(X,mode='reduced')
            Y = X.T.dot(Z)
            Z = X.dot(Y)
            Res = cp.multiply(zfdata-Z,datamask)
            res = cp.linalg.norm(Res,'fro')
            relres = res / datanrm
            ratio = res / res0
            reschg = cp.abs(1-res/res0)
            RR[iter_] = ratio
            # adjust alf
            if ratio >= 1.0:
                increment = max([0.1*alf,0.1*increment])
                X = cp.copy(X0)
                Y = cp.copy(Y0)
                Res = cp.copy(Res0)
                res = cp.copy(res0)
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
            Z = Z - cp.multiply(Z,datamask) + Zknown

        obj = objv[:iter_]

        return X, Y, [obj, RR, iter_, relres, reschg] 
            

# -----------------------TESTING----------------------------------------------

def plot_test_data(images2d):

    n = len(images2d)
    for num, img in enumerate(images2d):
        img = cp.asnumpy(img)
        plt.subplot(1,n,num+1)
        plt.imshow(img,cmap='gray',vmin=0,vmax=255)
    plt.show()

def make_test_data():
    #a = cp.array([cp.sin(i/3) for i in range(100)])
    #b = cp.array([cp.sin(i/3) for i in range(100)])
    a = cp.array([i for i in range(100)])    
    b = cp.array([i/3 for i in range(100)])    
    A = cp.outer(a,b)
    mask = cp.random.rand(A.shape[0],A.shape[1])
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    A_masked = cp.multiply(A,mask)    

    return A, A_masked, mask

def load_boat():

    im = imageio.imread(PICDIR+'/boat.png')
    im = cp.array(im)
    mask = cp.random.rand(im.shape[0],im.shape[1])
    mask[mask >= 0.7] = 1
    mask[mask < 0.7] = 0
    im_masked = cp.multiply(im,mask)    

    return im, im_masked, mask

if __name__ == '__main__':

    im, im_masked, mask = load_boat()
    #im, im_masked, mask = make_test_data()
    start_time = time.time()
    slv = LMaFit(im_masked)
    X, Y, out = slv.solve_mc()
    print('elapsed time: {}'.format(time.time()-start_time))
    plot_test_data([im, im_masked, mask, X.dot(Y)])

    print('Estimated rank : {}'.format(X.shape[1]))
