#!/usr/bin/python3
"""
same as fit-t2 so far, but with different start parameters....
should tune!
"""
import os
import sys
import glob
import math
import time
import numpy as np
import nibabel as nib


import multiprocessing
from scipy.optimize import curve_fit


from multicpu import apply_along_axis_parallel
from readprocpar import procparReader
from readfdf import fdfReader
from writenifti import niftiWriter
from argparse import ArgumentParser

# define some utility constants:
FIT_AXIS = 3
CORES = 4
CUT_AXIS = 1


def load_data_from_nifti(infile):

	data = nib.load(infile).get_fdata()

	# procpar should be supplied with nifti file as a hidden file: .[niftiname].procpar
	
	if infile.endswith('.nii'):
		procpar = '.'+infile[:-4]+'.procpar'
	elif infile.endswith('.nii.gz'):
		procpar = '.'+infile[:-7]+'.procpar'
	if os.path.isfile(procpar):
		pass
	else:
		raise
		sys.exit('No procpar found')

	return data, procpar

class FitT2():

	def __init__(self, data, procpar, fitmethod):

		self.data = data
		self.procpar = procpar
		self.fitmethod = fitmethod

		ppr = procparReader(procpar)
		self.ppdict = ppr.read()
		te = float(self.ppdict['te'])
		ne = int(self.ppdict['ne'])
		self.echo_times = [i*te for i in range(1,ne+1)] # echo times in seconds	

	def fit(self):
		"""
		Done by using np.apply_along_axes on custon defined fitting function.
		The fitting function uses scipy curve_fit. np.apply_along_axis is made to support parallel
		processing in custom function in 'multicpu' module.
		The actual function to be fitted can be chosen by the parameter 'fitmethod'
		Starting fit parameters should be fine-tuned by hand
		"""
		if self.fitmethod == 'two_parameter_monoexp':
		############################## MONO EXP WITHOUT CONST #####################################
			def fit1D(data1D, echo_times):		

				if all([v == 0 for v in data1D]):  # return zeros if data is all 0, which means masked
					return np.asarray([0.0, 0.0, 0.0])

				# Define callable exponential function. The c constant accounts for noise 
				def f(x, M0, T2):
					y = M0*np.exp(-x/T2)
					return y

				y = data1D
				x = np.asarray(echo_times)
				try:

					popt, pcov = curve_fit(f, x, y, p0=(10000.0, 0.01), check_finite=False, \
											bounds=([0.0, 0.0],[np.inf, 0.2]))
				except KeyboardInterrupt:
					raise		
				except:
					return np.asarray([0.0, 0.0, 0.0])
		 

				M0, T2 = popt
				res = y - f(x, M0, T2)
				ss_res = np.sum(res**2)
				ss_tot = np.sum((y-np.mean(y))**2)
				r_squared = 1 - (ss_res / ss_tot)

				return np.asarray([T2, M0, r_squared])
			print('fit-t2 : Fitting two parameter monoexponential ...')
			out = apply_along_axis_parallel(fit1D,\
											FIT_AXIS,\
											self.data,\
											self.echo_times,\
											cores=CORES,\
											cut_axis=CUT_AXIS)

			return out

		elif self.fitmethod == 'three_parameter_monoexp':
		############################## MONO EXP WITH CONST #####################################
			def fit1D(data1D, echo_times):

				if all([v == 0 for v in data1D]):  # return zeros if data is all 0, which means masked
					return np.asarray([0.0, 0.0, 0.0, 0.0])

				def f(x, M0, T2, c):
					y = M0*np.exp(-x/T2) + c
					return y

				y = data1D
				x = np.asarray(echo_times)
				try:

					popt, pcov = curve_fit(f, x, y, p0=(10000.0, 0.01, 50.0), check_finite=False, \
											bounds=([0.0, 0.0, 0.0],[np.inf, 0.2, 300.0]))
				except KeyboardInterrupt:
					raise
				except:
					return np.asarray([0.0, 0.0, 0.0, 0.0])
		 
				M0, T2, c = popt
				res = y - f(x, M0, T2, c)
				ss_res = np.sum(res**2)
				ss_tot = np.sum((y-np.mean(y))**2)
				r_squared = 1 - (ss_res / ss_tot)

				return np.asarray([T2, M0, c, r_squared])
			print('fit-t2 : Fitting three parameter monoexponential ...')
			out = apply_along_axis_parallel(fit1D,\
											FIT_AXIS,\
											self.data,\
											self.echo_times,\
											cores=CORES,\
											cut_axis=CUT_AXIS)
			return out

		elif self.fitmethod == 'four_parameter_duoexp':
		############################## DUAL EXP WITHOUT CONST #####################################
			def fit1D(data1D, echo_times):

				# define callable exponential function

				if all([v == 0 for v in data1D]):  # return zeros if data is all 0, which means masked
					return np.asarray([0.0, 0.0, 0.0, 0.0, 0.0])

				def func(x, m0, t2s, t2l, f):
					y = m0*(f * np.exp(-x / t2s) + (1-f) * np.exp(-x / t2l))
					return y

				y = data1D
				x = np.asarray(echo_times)
				try:	
					popt, pcov = curve_fit(func, x, y, p0=(10000.0, 0.01, 0.07, 0.5), check_finite=False,\
										bounds=([0.0, 0.0, 0.0, 0.0],[np.inf, 0.1, 0.2, 1.0]))
				except KeyboardInterrupt:
					raise
				except:
					return np.asarray([0.0, 0.0, 0.0, 0.0, 0.0])
		 
				m0, t2s, t2l, f = popt
				res = y - func(x, m0, t2s, t2l, f)
				ss_res = np.sum(res**2)
				ss_tot = np.sum((y-np.mean(y))**2)
				r_squared = 1 - (ss_res / ss_tot)

				return np.asarray([m0, t2s, t2l, f, r_squared])
			print('fit-t2 : Fitting four parameter dual exponential ...')
			out = apply_along_axis_parallel(fit1D,\
											FIT_AXIS,\
											self.data,\
											self.echo_times,\
											cores=CORES,\
											cut_axis=CUT_AXIS)

			return out

def save_results_to_nifti(res, nifti, fitmethod, outname=None):

	if nifti.endswith('.nii'):
		nifti_base = nifti[:-4]
	elif nifti.endswith('.nii.gz'):
		nifti_base = nifti[:-7]

	if fitmethod == 'two_parameter_monoexp':

		t2map = out.take(indices=0, axis=FIT_AXIS)
		m0map = out.take(indices=1, axis=FIT_AXIS)
		R2map = out.take(indices=2, axis=FIT_AXIS)

		output_name_list = []
		output_basename_list = ['t2map', 'm0map', 'R2map']
		for item in output_basename_list:
			output_name_list.append(str(nifti_base)+'_'+item)
		output_list = [t2map, m0map, R2map]

		for num, item in enumerate(output_name_list):
			writer = niftiWriter(procpar, output_list[num])
			writer.write(item)

	if fitmethod == 'three_parameter_monoexp':

		t2map = out.take(indices=0, axis=FIT_AXIS)
		m0map = out.take(indices=1, axis=FIT_AXIS)
		constmap = out.take(indeices=2, axis=FIT_AXIS)
		R2map = out.take(indices=3, axis=FIT_AXIS)

		output_name_list = []
		output_basename_list = ['t2map', 'm0map', 'constmap','R2map']
		for item in output_basename_list:
			output_name_list.append(str(nifti_base)+'_'+item)
		output_list = [t2map, m0map, R2map]

		for num, item in enumerate(output_name_list):
			writer = niftiWriter(procpar, output_list[num])
			writer.write(item)

	if fitmethod == 'four_parameter_duoexp':

		
		m0map = out.take(indices=0, axis=FIT_AXIS)
		t2smap = out.take(indices=1, axis=FIT_AXIS)
		t2lmap = out.take(indices=2, axis=FIT_AXIS)
		fractmap = out.take(indeices=3, axis=FIT_AXIS)
		R2map = out.take(indices=4, axis=FIT_AXIS)

		output_name_list = []
		output_basename_list = ['m0map', 't2smap', 't2lmap', 'fractmap', 'R2map']
		for item in output_basename_list:
			output_name_list.append(str(nifti_base)+'_'+item)
		output_list = [t2map, m0map, R2map]

		for num, item in enumerate(output_name_list):
			writer = niftiWriter(procpar, output_list[num])
			writer.write(item)

if __name__ == '__main__':
	
	parser = ArgumentParser(description='Fit T2 to files in Nifti format')
	parser.add_argument('input',\
						help='input Nifti',\
						metavar='INPUT.nii')
	parser.add_argument('output',\
						help='output basename',
						metavar='OUTPUT',\
						nargs='?',\
						default=None)
	parser.add_argument('--fitmethod',\
						dest='fitmethod',\
						help='fitting method',\
						default='two_parameter_monoexp')
	args = parser.parse_args()

	if args.input.endswith('.gz') or args.input.endswith('.nii'):
		pass
	else:
		sys.exit('Error: Input is not a nifti')

	data, procpar = load_data_from_nifti(args.input)
	fitter = FitT2(data, procpar, args.fitmethod)
	out = fitter.fit()
	save_results_to_nifti(out, args.input, args.fitmethod, args.output)
