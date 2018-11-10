#!/usr/bin/python3

import os
import glob
import matplotlib.pyplot as plt
import nmrglue as ng
from argparse import ArgumentParser
import numpy as np

class spulsPlotter():
	
	def __init__(self, spulsdir, no_channel_add=True, normalize=True):

		self.spulsdir = spulsdir	
		self.fft = []
		self.no_channel_add = no_channel_add
		(pardict, fid) = ng.varian.read(spulsdir)  # read in the data
		ppdict = pardict['procpar']
		self.sw = float(ppdict['sw']['values'][0])  # get receive spectral width
		self.averages = int(ppdict['nt']['values'][0])
		self.normalize = normalize
		
		print('fid type: '+str(type(fid)))
		print('fid shape: '+str(fid.shape))

		for i in range(fid.shape[0]):
			self.fft.append(ng.proc_base.fft(fid[i,:]))
		self.fft = np.asarray(self.fft)

		if not self.no_channel_add:
	
			self.fft_combined = np.mean(np.absolute(self.fft), axis=0)

		else:
			pass
			
		print('self fft shape : '+str(self.fft.shape))

		
		

	def plot(self):

		if not self.no_channel_add:

			x_axis = np.linspace(0, self.sw, self.fft.shape[1])
			y_axis = self.fft_combined

			plt.plot(x_axis, y_axis)
			plt.xlim(0, self.sw)
			plt.show()

		else:

			x_axis = np.linspace(0, self.sw, self.fft.shape[1])
			plt.subplots(self.fft.shape[0],1)
			for i in range(self.fft.shape[0]):
				plt.subplot(self.fft.shape[0],1,i+1)
				y_axis = np.absolute(self.fft[i])
				plt.plot(x_axis, y_axis)
				plt.xlim(0, self.sw)
				plt.ylabel('ch'+str(i+1))

			plt.show()

	def get_snr(self, channel):

		if self.normalize:
			abs_data = np.absolute(self.fft[channel,:]) / self.averages
		else:
			pass
		noise = np.mean(abs_data[0:300])
		bl_corrected = ng.process.proc_bl.cbf(abs_data, last=30)

		i = 980
		j = 1100
		integral = np.trapz(bl_corrected[i:j], dx=1)
		print('integral : '+str(integral))
		print('noise: '+str(noise))
		
		print('SNR :'+str(integral/noise))
		print('normalized SNR : '+str(integral/noise/np.sqrt(self.averages)))
		x_axis = np.linspace(0, self.sw, self.fft.shape[1])
		plt.plot(x_axis, bl_corrected)
		plt.axvline(x=i*self.sw/self.fft.shape[1], color='red')
		plt.axvline(x=j*self.sw/self.fft.shape[1], color='red')
		plt.xlim(0, self.sw)
		plt.show()
		


		

if __name__ == '__main__':
	
	parser = ArgumentParser()
	parser.add_argument('-d', dest='directory', help='spuls.fid directory')
	parser.add_argument('--noadd', dest='no_channel_add', \
						help='option to plot rx channels separately', default='True')
	args = parser.parse_args()
	splotter = spulsPlotter(args.directory, args.no_channel_add)
	#splotter.plot()

	splotter.get_snr(3)
