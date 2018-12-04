#!/usr/local/bin/python3.6

from cvxpy import *
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
"""
Collection of solvers for low rank matrix completion
"""


def svd_complete():

    pass

def nuclear_norm_solve(hankel_lr):

    hankel_filled = NuclearNormMinimization(max_iters=50).fit_transform(hankel_lr)

    return hankel_filled

def svt_solve(A, mask, tau=1000000, delta=None, epsilon=1e-4, max_iter=10000):

    Y = np.zeros_like(A)
    if tau == None:
        tau = 5 * np.sum(A.shape) / 2
        print('tau : {}'.format(tau))
    if delta == None:
        delta = 1.2 * np.prod(A.shape) / np.sum(mask)

    for _ in range(max_iter):

        U, S, V = np.linalg.svd(Y, full_matrices=False)
        S = np.maximum(S - tau, 0)
        X = np.linalg.multi_dot([U, np.diag(S), V])
        Y = Y + delta*mask*(A-X)

        rel_recon_error = np.linalg.norm(mask*(X-A))/np.linalg.norm(mask*A)

        if _ % 100 == 0:
            print(rel_recon_error)
            print(S.shape)
            pass
        if rel_recon_error < epsilon:
            break

    return X

class SVTSolver():

    def __init__(self,A,tau=None, delta=None, epsilon=1e-4, max_iter=10000):

        pass

    def solve(self):

        pass

def main():

    a = np.array([i for i in range(100)])
    b = np.array([i/2 for i in range(100)])

    A = np.outer(a,b)
    M = np.random.rand(A.shape[0],A.shape[1])
    # mask matrix
    M[M < 0.8] = 0
    M[M >= 0.2] = 1
    A_m = np.multiply(A,M)
    A_sol = svt_solve(A_m, M)
    #print(A_sol-A)
    print((A-A_sol)/A)
    print('----------------------------')
    print(A_sol)
    #print(A_sol)
    print(np.linalg.matrix_rank(A_sol))
    print(np.linalg.matrix_rank(A))
    plt.subplot(1,3,1)
    plt.title('solved')
    plt.imshow(A_sol)
    plt.subplot(1,3,2)
    plt.title('original')
    plt.imshow(A)   
    plt.subplot(1,3,3)
    plt.title('orig w missing')
    plt.imshow(A_m)   
    plt.show()

if __name__ == '__main__':

    main()
