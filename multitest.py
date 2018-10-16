#!/usr/bin/python3

import multiprocessing
import glob
import scipy as sp
from scipy import stats
import nibabel as nib
import matplotlib.pyplot as plt
from optparse import OptionParser
from datetime import datetime

def add(img_data):

	return sp.mean(img_data), sp.mean(img_data) * 2	

def lin_fit(img_data):
# actual fitting function
# linear fitting after taking natural logarithm
	te = 8
	ne = 8
	TE_data = []
	for i in range(int(float(ne))):
		TE_data.append((i+1) * int(float(te)))
	ln_data = sp.log(img_data[:])
	slope, intercept, r_value, p_value, std_err = stats.linregress(TE_data, ln_data)	
	return -1/slope, r_value


def fit(in_data, output, num):
# basically numpy apply_along_axis for parts of data
	axis = 3
	out_part = sp.apply_along_axis(add, axis, in_data)
	output.put((num, out_part))


def multifit(data):
# cuts nifti to smaller chunks and distributes to CPUs
	workers = 4
	jobs = []
	output = multiprocessing.Queue()
	part = int(128 / workers)
	for i in range(workers):
		data_chunk = data[i*part:(i+1)*part,:,:,:]
		p = multiprocessing.Process(target=fit, args=(data_chunk, output, i))
		jobs.append(p)
		p.start()
	print("waiting for job")
	out = [output.get() for i in jobs]
	
	for job in jobs:
		job.join()
	print("all jobs done")
	out.sort()
	out = [r[1] for r in out]
	out = sp.concatenate(out,0)
	t2 = out[:,:,:,0]
	r2_err = out[:,:,:,1]
	return [t2, r2_err]
	


def main():
	

	startTime = datetime.now()

	parser = OptionParser()
	[opts, args] = parser.parse_args()
	nifti = nib.load("mems_20180222_01__8NE_.nii")
	data = nifti.get_data()
	pars = {"te " : 8, "ne " : 8}
	#plt.imshow(data[:,:,15,1], cmap="gray")
	[t2_fit_data, r2_err_data] = multifit(data)
	out_t2 = nib.Nifti1Image(t2_fit_data, nifti.affine, nifti.header)
	out_err = nib.Nifti1Image(r2_err_data, nifti.affine, nifti.header)
	nib.save(out_t2, "t2_test.nii")
	nib.save(out_err, "t2_err_test.nii")
	print("time:  " + str(datetime.now() - startTime))
if __name__ == "__main__":
	main()
