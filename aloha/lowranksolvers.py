#!/usr/local/bin/python3.6

from cvxpy import *
import copy
import numpy as np
import matplotlib.pyplot as plt
import sys

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
"""
Collection of solvers for low rank matrix completion
"""

class LMaFit():

    def __init__(self):
        pass
    def solve(self):
        pass

class SVTSolver():
    """
    Matrix completion by singular value soft thresholding
    """
    def __init__(self,A,tau=None, delta=None, epsilon=1e-4, max_iter=1000):

        self.A = A
        self.Y = np.zeros_like(A)
        self.max_iter = max_iter
        self.epsilon = epsilon
        mask = copy.deepcopy(A)
        mask[mask != 0] = 1
        self.mask = mask 
        if tau == None:
            self.tau = 5*np.sum(A.shape)/2
        else:
            self.tau = tau
        if delta == None:
            self.delta = 1.2*np.prod(A.shape)/np.sum(self.mask)
        else:
            self.delta = delta

    def solve(self):
        """Main iteration, returns the completed matrix"""
        for _ in range(self.max_iter):

            U, S, V = np.linalg.svd(self.Y, full_matrices=False)
            S = np.maximum(S-self.tau, 0)
            X = np.linalg.multi_dot([U, np.diag(S), V])
            self.Y = self.Y + self.delta*self.mask*(self.A-X)
        
            rel_recon_error = np.linalg.norm(self.mask*(X-self.A)) / \
                                np.linalg.norm(self.mask*self.A)

            if _ % 100 == 0:
                sys.stdout.flush()
                print(rel_recon_error)
                pass
            if rel_recon_error < self.epsilon:
                break
        return X
#-----------------------------TESTING FUNCTIONS--------------------------
def plot_test_results():
    pass
def make_test_data():

    a = np.array([i for i in range(100)])
    b = np.array([i/2 for i in range(100)])

    A = np.outer(a,b)
    A_comp = np.vectorize(complex)(A,A)
    M = np.random.rand(A.shape[0],A.shape[1])
    # mask matrix
    M[M < 0.8] = 0
    M[M >= 0.2] = 1
    A_m = np.multiply(A,M)

    return A, A_m, M

def main():

    A, A_m, M = make_test_data()


    return
    A_sol = SVTSolver(A_m, tau=500000,delta=1).solve()
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
