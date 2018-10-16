#!/usr/bin/python3


import numpy as np
import multiprocessing


def apply_along_axis_parallel(func1d, axis, arr, funcargs, cores=None, cut_axis=-1):
	"""
	Basically numpy.apply_along_axis, but for multiple cores.
	cores = Number of CPU cores to use. if "None" use maxcores-2
	cut_axis = Axis to cut the array along for multiprocessing
	"""
	def aaap_helper(func1d, in_data, funcargs, output, num):
		"""
		Helper for arranging outputs of numpy.apply_along_axis with Queue.put()
		"""
		out_part = np.apply_along_axis(func1d, axis, in_data, funcargs)
		output.put((num, out_part))

	if cores == None:
		cores = multiprocessing.cpu_count()-2
	elif cores > multiprocessing.cpu_count():
		print('Warning! More workers specified than CPU threads ')


	try: # if axis length can be divided by cores it's OK. Chunks is a list
		data_chunks = np.split(arr, cores, cut_axis)
	except: # if axis length cannot be divided, then leave a remainder
		data_chunks = np.split(arr, cores-1, cut_axis)

	jobs = []
	output = multiprocessing.Queue()

	for i in range(cores):
		
		p = multiprocessing.Process(target=aaap_helper, \
					args=(func1d, data_chunks[i], funcargs, output, i))
		jobs.append(p)
		p.start()

	out = [output.get() for i in jobs]
	
	for job in jobs:
		job.join()
	print("all jobs done")
	out.sort()

	out = [r[1] for r in out]  # why is it here????


	out = np.concatenate(out,cut_axis)
	print('multicpu out.shape after concatenate: '+str(out.shape))

	return out







