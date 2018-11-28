#!/usr/local/bin/python3.6

import numpy as np

"""
Functions for handling Hankel matrices in various ALOHA implementations

"""

def decompose_hankel(matrix, n,m=0,o=0,p=0,q=0,r=0):

    if m == 0 and r == 0: # one level decompoosition

        elements = np.concatenate(matrix[:,0],matrix[-1,1:],axis=0)
        
        return elements

    elif m != 0 and r == 0: # two level decomposition

        cols = matrix[]

        pass

    elif m != 0 and r != 0:

    
if __name__ == '__main__':

    pass
        

    
