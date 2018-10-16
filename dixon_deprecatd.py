#!/usr/bin/python3

import os
import sys
import glob
import math
import time
import numpy as np
import nibabel as nib
from argparse import ArgumentParser
sys.path.append('/home/david/bin')
sys.path.append('/home/david/dev/common')
from readprocpar import procparReader
from readfdf import fdfReader
from writenifti import niftiWriter
from dixonpreproc import dixonPreproc
from dixonproc import dixonProc

def main(study_dir, freq):

	dpp = dixonPreproc(study_dir)
	dpp.run()
    if freq == None:

        dp = dixonProc(study_dir)
        dp.least_squares_fit()

    else:

        dp = dixonProc

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-s','--study_dir',dest='study_dir')
	parser.add_argument('-f','--freq',dest='freq',nargs='*',\
                        metavar = ' freq1 freq2', default=None)
	parser.add_argument('--full',dest='full', action=store_true, default=False)

	args = parser.parse_args()
    print(args)
    #main(args.study_dir, args.freq)
