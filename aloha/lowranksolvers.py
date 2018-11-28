#!/usr/local/bin/python3.6

from cvxpy import *
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute
"""
Collection of solvers for low rank matrix completion
"""


def svd_complete():

    pass

def nuclear_norm_solve(hankel_lr):

    hankel_filled = NuclearNormMinimization(max_iters=50).fit_transform(hankel_lr)

    return hankel_filled
