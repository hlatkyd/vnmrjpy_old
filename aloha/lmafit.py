#!/usr/bin/python3

import copy
import numpy as np
import matplotlib.pyplot as plt
# translated from lmafit_mc_adp.m by Mark Crovella December 2014
# omitting many options and just implementing the core functionality
# for documentation on lmafit see http://lmafit.blogs.rice.edu/
#
# Note: this is very useful:
# http://wiki.scipy.org/NumPy_for_Matlab_Users
#

MAXIT = 5000
TOL = 1e-5
K = 15

def lmafit_new(data,k=K):

    pass

class LMaFit():
    """ LMaFit.m rewrite"""
    def __init__(self,zfdata,k,opts=None):

        self.zerofilled_data = zfdata
        self.k = k
        self.tol = 1.25e-5
        self.maxit = 5000
        self.iprint = 0
        
    def solve_mc(self):
        
        k = self.k
        datanrm = np.max([1,np.linalg.norm(self.zerofilled_data)])
        objv = np.zeros(self.maxit)
        RR = np.ones(self.maxit)
        # init
        Z = copy.deepcopy(self.zerofilled_data)
        (m,n) = self.zerofilled_data.shape
        X = np.zeros((m,k))
        Y = np.eye(k,n)
        Res = self.zerofilled_data
        res = datanrm
        print('first res {}'.format(res))
        reschg_tol = 0.5*self.tol
        # parameters for alf
        alf = 0
        increment = 1
        itr_rank = 0
        minitr_reduce_rank = 5
        maxitr_reduce_rank = 50

        for iter_ in range(self.maxit):
            print('alf : {}'.format(alf))
            itr_rank += 1
            X0 = copy.deepcopy(X)
            Y0 = copy.deepcopy(Y)
            Res0 = copy.deepcopy(Res)
            res0 = copy.deepcopy(res)
            alf0x = alf
            Z0 = copy.deepcopy(Z)
            X = Z.dot(Y.T)
            X, R = np.linalg.qr(X)
            print('X shape {}'.format(X.shape))
            Y = X.T.dot(Z)
            Z = X.dot(Y)
            # TODO  is this wrong??? 
            Res = self.zerofilled_data - Z
            res = np.linalg.norm(Res[self.zerofilled_data != 0])
            relres = res / datanrm
            print('relres : {}'.format(relres))
            ratio = res / res0
            print('ratio : {}'.format(ratio))
            reschg = np.abs(1-res/res0)
            RR[iter_] = ratio
            # adjust alf
            if ratio >= 1.0: # WTF IS THIS
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
                                    or (relres < self.tol))):
                print('Stopping crit achieved')
                break

            Z = Z + alf*Res

        obj = objv[:iter_]

        return X, Y, [obj, RR, iter_, relres, reschg] 
            

def lmafit_mc_adp(m,n,k,Known,data,opts=None):
    """
    Output:
           X --- m x k matrix
           Y --- k x n matrix
         Out --- output information
     Input:
        m, n --- matrix sizes
           k --- rank estimate
       Known is a 2xL ndarray holding indices of known elements 
        data --- values of known elements in a 1D row vector
        opts --- option structure (not used)
    """
    L = len(data)
    tol = 1.25e-4
    maxit = 500
    iprint = 0
    reschg_tol = 0.5*tol
    datanrm = np.max([1.0,np.linalg.norm(data)])
    objv = np.zeros(maxit)
    RR = np.ones(maxit)
    if iprint == 1:
        print('Iteration: ')
    if iprint == 2:
        print('\nLMafit_mc: \n')

    # initialize: make sure the correctness of the index set and data
    data[data==0]=np.spacing(1)
    data_tran = False
    Z = np.zeros((m,n))
    Z[Known] = data

    if m>n:
        tmp = m
        m = n
        n = tmp
        Z = Z.T
        Known = np.nonzero(Z)
        data = Z[Known]
        data_tran = True

    # assuming no inital solutions provided
    X = np.zeros((m,k))
    Y = np.eye(k,n)
    Res = data
    res = datanrm

    # parameters for alf
    alf = 0
    increment = 1
    itr_rank = 0
    minitr_reduce_rank = 5
    maxitr_reduce_rank = 50

    for iter in range(maxit):
        itr_rank += 1
        Xo = X
        Yo = Y
        Res0 = Res
        res0 = res
        alf0x = alf
        # iterative step
        # Zfull option only
        Zo = Z
        X = Z.dot(Y.T)
        X, R = np.linalg.qr(X)
        Y = X.T.dot(Z)
        Z = X.dot(Y)
        Res = data - Z[Known]
        #
        res = np.linalg.norm(Res)
        relres = res/datanrm
        ratio = res/res0
        reschg = np.abs(1-res/res0)
        RR[iter] = ratio
        # omitting rank estimation
        # adjust alf
        if ratio>=1.0:
            increment = np.max([0.1*alf, 0.1*increment])
            X = Xo
            Y = Yo
            Res = Res0
            res = res0
            relres = res/datanrm
            alf = 0
            Z = Zo
        elif ratio>0.7:
            increment = max(increment, 0.25*alf)
            alf = alf+increment;
    
        if iprint==1:
            print('{}'.format(iter))
        if iprint==2:
            print('it: {} rk: (none), rel. {} r. {} chg: {} alf: {} inc: {}\n'\
                    .format(iter, k, relres,ratio,reschg,alf0x,increment))

        objv[iter] = relres

        # check stopping
        if ((reschg < reschg_tol) and ((itr_rank > minitr_reduce_rank) \
                                    or (relres < tol))):
            break
    
        Z[Known] = data + alf*Res

    if iprint == 1:
        print('\n')

    if data_tran:
        tX = X
        X = Y.T
        Y = tX.T

    obj = objv[:iter]

    return X, Y, [obj, RR, iter, relres, reschg]

# -----------------------TESTING----------------------------------------------

def plot_test_data(images2d):

    n = len(images2d)
    for num, img in enumerate(images2d):
        plt.subplot(1,n,num+1)
        plt.imshow(img,cmap='gray')
    plt.show()

def make_test_data():

    #a = np.array([np.sin(i/3) for i in range(100)])
    #b = np.array([np.sin(i/3) for i in range(100)])
    a = np.array([i for i in range(100)])    
    b = np.array([i/3 for i in range(100)])    

    A = np.outer(a,b)
    mask = np.random.rand(A.shape[0],A.shape[1])
    mask[mask >= 0.8] = 1
    mask[mask < 0.8] = 0
    A_masked = np.multiply(A,mask)    

    return A, A_masked, mask

def lmafit_input_preproc(A_masked):
    """Preprocess masked data for LMaFit"""
    (m,n) = A_masked.shape
    vec1d = np.reshape(A_masked,(A_masked.size),order='f')
    known = np.nonzero(vec1d)[0]
    data = vec1d[known]
    return m, n, known, data

if __name__ == '__main__':

    A, A_masked, mask = make_test_data()
    #k = 50
    #slv = LMaFit(A_masked,k)
    #X, Y, out = slv.solve_mc()
    #print(X.shape)
    #print(Y.shape)
    lmafit_new(A_masked)
    #plot_test_data([A, A_masked, mask, X.dot(Y)])
    #m, n, known, data = lmafit_input_preproc(A_masked)
    #k = m
    #X, Y, out = lmafit_mc_adp(m,n,k, known, data)


