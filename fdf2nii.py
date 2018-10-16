#!/usr/bin/python3

"""
Script to convert .fdf to .nii files. The input can be 2D or 3D. Nifit affines are calculated based
on procpar file, which should be included with all the .fdf files in the input .img directory.

"""
import os
import sys
import glob
import numpy as np

from shutil import copy2
from argparse import ArgumentParser
from writenifti import niftiWriter
from readprocpar import procparReader
from readfdf import fdfReader

def fdf2nii(indir, out, saveprocpar):

	if indir.endswith('.img'):
		if out == None:
			out = str(indir[:-4].split('/')[-1]+'.nii')
			out_ppname = '.'+out[:-4]+'.procpar'
		else:
			out_ppname = '.' + out + '.procpar' 
	else:
		print('Please give .img as input directory')
		return
	try:
		procpar = indir + '/procpar'
	except:
		print('procpar not found, quitting ...')
		return
	rdr = fdfReader(indir, out)
	(hdr, data) = rdr.read()

	writer = niftiWriter(procpar,data)
	writer.write(out)
	if saveprocpar:
		
		copy2(procpar, out_ppname)
		print('procpar saved as hidden: '+str(out_ppname))

# utility for test, shell call
if __name__ == '__main__':
	
	parser = ArgumentParser(description='Convert fdf files from .img directory to Nifti format')
	parser.add_argument('input',\
						help='input .img directory',\
						metavar='INPUTDIR.img')
	parser.add_argument('output',\
						help='output .nii.gz file',
						metavar='OUTPUT.nii.gz',\
						nargs='?',\
						default=None)
	parser.add_argument('--saveprocpar',\
						dest='saveprocpar',\
						help='Boolean, dfaults to True',\
						default=True)
	args = parser.parse_args()

	fdf2nii(args.input, args.output, args.saveprocpar)
