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
sys.path.append('/home/david/dev/vnmrjpy')
from readprocpar import procparReader
from readfdf import fdfReader
from writenifti import niftiWriter
from dixonpreproc import dixonPreproc
from dixonproc import dixonProc

def main(study_dir, points, infreq):

    dpp = dixonPreproc(study_dir)
    dpp.run()

    if infreq != None:
        freq = [float(infreq[0]), float(infreq[1])]

        dp = dixonProc(study_dir, freq=freq)
        #dp.least_squares_fit(points)
        dp.ideal_fit(100)

    elif infreq == None:
        dp = dixonProc(study_dir)
        #dp.least_squares_fit(points)
        dp.ideal_fit(100)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s','--study_dir',dest='study_dir')
    parser.add_argument('-p','--points',dest='points', default=None)
    parser.add_argument('-f','--freq',metavar=' freq1 freq2', nargs='*',\
                        default=None)
    parser.add_argument('--full',dest='fullproc', action='store_true',\
                        default=None)
    args = parser.parse_args()
    main(args.study_dir, args.points, args.freq)
