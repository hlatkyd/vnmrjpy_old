#!/usr/bin/python3

import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from readprocpar import procparReader
from readfid import fidReader
from writenifti import niftiWriter

class recon():
	"""
	Class for various MRI, NMR sequence reconstruction
	
	INPUT:  procpar - procpar file

			data - data in numpy.ndarray[blocks, traces*np] format

	METHODS: recon()

			return kspace=numpy.ndarray([receivers,volumes,slice,phase,read]), imspace = --same--
	"""
	def __init__(self, procpar, data):
		ppr = procparReader(procpar)	
		self.ppdict = ppr.read()
		self.data = data
		seqfil = self.ppdict['seqfil']
		seqcon = self.ppdict['seqcon']

	def recon(self):
		# -------initializing parameters----------------------
		seqfil = self.ppdict['seqfil']
		seqcon = self.ppdict['seqcon']
		if self.ppdict['rcvrs'] == 'yyyy':		
			rcvrs = 4
		elif self.ppdict['rcvrs'] == 'y':
			rcvrs = 1
		else:
			print('Something happened: cant find number of receivers, quitting...')
			return

		#-----------------------------------------------------------------------
		#						2D GEMS reconstruction
		#-----------------------------------------------------------------------
		if seqfil == 'gems':

			print('\nReconstructing gems...\n')
			interleaved = int(self.ppdict['sliceorder'])
			nread = int(int(self.ppdict['np'])/2)
			nphase = int(self.ppdict['nv'])
			nslice = int(self.ppdict['ns'])
			#------------------shaping the k-space---------------------------
			if seqcon == 'nccnn':
				shape = (rcvrs,nread,nslice,nphase)
				kspace = np.vectorize(complex)(self.data[:,0::2],self.data[:,1::2])
				kspace = np.reshape(kspace,shape,order='C')
				if interleaved:
					n = int(kspace.shape[2]/2)
					c = np.zeros(shape,dtype=complex)
					c[:,:,1::2,:] = kspace[:,:,n:,:]
					c[:,:,0::2,:] = kspace[:,:,:n,:]
					kspace = c
				kspace = np.swapaxes(kspace,2,1)
				kspace = np.expand_dims(kspace,1)
			
			#---------making the image with inverse fft--------------------
			imspace = np.fft.ifft2(kspace,axes=(3,4),norm='ortho')
			imspace = np.fft.fftshift(imspace,axes=(3,4))

			return kspace, imspace
		#-----------------------------------------------------------------------
		#						2D SEMS reconstruction
		#-----------------------------------------------------------------------
		if seqfil == 'sems':

			print('\nReconstructing sems...\n')
			interleaved = int(self.ppdict['sliceorder'])
			nread = int(int(self.ppdict['np'])/2)
			nphase = int(self.ppdict['nv'])
			nslice = int(self.ppdict['ns'])
			#------------------shaping the k-space---------------------------
			if seqcon == 'nccnn':

				shape = (rcvrs,nread,nslice,nphase)
				kspace = np.vectorize(complex)(self.data[:,0::2],self.data[:,1::2])
				kspace = np.reshape(kspace,shape,order='C')
				if interleaved:
					n = int(kspace.shape[2]/2)
					c = np.zeros(shape,dtype=complex)
					c[:,:,1::2,:] = kspace[:,:,n:,:]
					c[:,:,0::2,:] = kspace[:,:,:n,:]
					kspace = c
				kspace = np.swapaxes(kspace,2,1)
				kspace = np.expand_dims(kspace,1)

			elif seqcon == 'ncsnn':
							
				shape = (rcvrs,nread,nslice,nphase)				
				kspace = np.vectorize(complex)(self.data[:,0::2],self.data[:,1::2])
				preshape = (rcvrs,int(kspace.shape[0]/rcvrs),-1)	
				kspace = np.reshape(kspace,preshape,order='F')
				kspace = np.reshape(kspace,shape,order='C')
				if interleaved:
					n = int(kspace.shape[2]/2)
					c = np.zeros(shape,dtype=complex)
					c[:,:,1::2,:] = kspace[:,:,n:,:]
					c[:,:,0::2,:] = kspace[:,:,:n,:]
					kspace = c
				kspace = np.swapaxes(kspace,2,1)
				kspace = np.expand_dims(kspace,1)
			
			#---------making the image with inverse fft--------------------
			imspace = np.fft.ifft2(kspace,axes=(3,4),norm='ortho')
			imspace = np.fft.fftshift(imspace,axes=(3,4))

			return kspace, imspace
		#-----------------------------------------------------------------------
		#						2D FSE reconstruction - in progress
		#-----------------------------------------------------------------------
		if seqfil == 'im2Dfse':
			pass
		#-----------------------------------------------------------------------
		#						GE3D reconstruction
		#-----------------------------------------------------------------------
		if seqfil == 'ge3d':

			print('\nReconstructing ge3d...\n')

			nread = int(int(self.ppdict['np'])/2)
			nphase = int(self.ppdict['nv'])
			nphase2 = int(self.ppdict['nv2'])
			#------------------shaping the k-space---------------------------
			if seqcon == 'nccsn':
				shape = (rcvrs,nread,nphase2,nphase)
				kspace = np.vectorize(complex)(self.data[:,0::2],self.data[:,1::2])
				print('kspace before reshape : '+str(kspace.shape))
				preshape = (rcvrs,int(kspace.shape[0]/rcvrs),-1)	
				#kspace = np.reshape(kspace,preshape,order='C')
				print('kspace after preshape : '+str(kspace.shape))
				kspace = np.reshape(kspace,shape,order='F')

				#kspace = np.swapaxes(kspace,2,1)
				kspace = np.expand_dims(kspace,1)
			
			#---------making the image with inverse fft--------------------
			imspace = np.fft.ifftn(kspace,axes=(2,3,4),norm='ortho')
			imspace = np.fft.fftshift(imspace,axes=(2,3,4))

			return kspace, imspace
		#-----------------------------------------------------------------------
		#						MGE3D reconstruction
		#-----------------------------------------------------------------------
		if seqfil == 'mge3d':

			print('\nReconstructing mge3d...\n')

			nread = int(int(self.ppdict['np'])/2)
			nphase = int(self.ppdict['nv'])
			nphase2 = int(self.ppdict['nv2'])
			ne = int(self.ppdict['ne'])

			#------------------shaping the k-space---------------------------
			if seqcon == 'cccsn':
				shape = (nphase2,rcvrs,nphase,ne,nread)
				kspace = np.vectorize(complex)(self.data[:,0::2],self.data[:,1::2])
				print('kspace before reshape : '+str(kspace.shape))
				preshape = (rcvrs,int(kspace.shape[0]/rcvrs),-1)	
				#kspace = np.reshape(kspace,preshape,order='C')
				print('kspace after preshape : '+str(kspace.shape))
				kspace = np.reshape(kspace,shape,order='C')

				kspace = np.swapaxes(kspace,0,1)
				kspace = np.swapaxes(kspace,3,1)
				#kspace = np.expand_dims(kspace,1)
			
			#---------making the image with inverse fft--------------------
			kspace = np.fft.fftshift(kspace,axes=(2,3,4))
			imspace = np.fft.ifftn(kspace,axes=(2,3,4), norm='ortho')
			imspace = np.fft.ifftshift(imspace,axes=(2,3,4))

			return kspace, imspace
		#-----------------------------------------------------------------------
		#						MPRAGE3D reconstruction - in progress
		#-----------------------------------------------------------------------
		if seqfil == 'mprage':

			print('\nReconstructing mprage...\n')

			nread = int(int(self.ppdict['np'])/2)
			nphase = int(self.ppdict['nv'])
			nphase2 = int(self.ppdict['nv2'])

			etl = int(self.ppdict['etl'])
			segments = int(int(self.ppdict['nv'])/etl)
			#------------------shaping the k-space---------------------------
			if seqcon == 'nccsn':
				shape = (rcvrs,nread,nphase2,nphase)
				kspace = np.vectorize(complex)(self.data[:,0::2],self.data[:,1::2])

				preshape = (rcvrs,int(kspace.shape[0]/rcvrs),-1)	
				kspace = np.reshape(kspace,preshape,order='C')

				kspace = np.reshape(kspace,shape,order='F')

				#kspace = np.swapaxes(kspace,2,1)
				kspace = np.expand_dims(kspace,1)
				
			#---------making the image with inverse fft--------------------
			kspace = np.fft.ifftshift(kspace,axes=(2,3,4))
			imspace = np.fft.ifftn(kspace,axes=(2,3,4),norm='ortho')
			#imspace = np.fft.fftshift(imspace,axes=(2,3,4))

			return kspace, imspace


#---------------------------------------------------------------------------------------------------
#                                 END OF Standard sequence RECONSTRUCTION
#---------------------------------------------------------------------------------------------------
		#-----------------------------------------------------------------------
		#						GE3D elliptical reconstruction
		#-----------------------------------------------------------------------
		if seqfil == 'ge3d_elliptical':

			print('\nReconstructing ge3d elliptical...\n')

			nread = int(int(self.ppdict['np'])/2)
			nphase = int(self.ppdict['nv'])
			nphase2 = int(self.ppdict['nv2'])
			#------------------shaping the k-space---------------------------
			if seqcon == 'nccsn':
				shape = (rcvrs,nread,nphase2,nphase)
				kspace = np.vectorize(complex)(self.data[:,0::2],self.data[:,1::2])
				print('kspace before reshape : '+str(kspace.shape))
				preshape = (rcvrs,int(kspace.shape[0]/rcvrs),-1)	
				#kspace = np.reshape(kspace,preshape,order='C')
				print('kspace after preshape : '+str(kspace.shape))
				kspace = np.reshape(kspace,shape,order='F')

				#kspace = np.swapaxes(kspace,2,1)
				kspace = np.expand_dims(kspace,1)
			
			#---------making the image with inverse fft--------------------
			imspace = np.fft.ifftn(kspace,axes=(2,3,4),norm='ortho')
			imspace = np.fft.fftshift(imspace,axes=(2,3,4))

			return kspace, imspace

#---------------------------------------------------------------------------------------------------

#                                 END OF RECONSTRUCTION

#---------------------------------------------------------------------------------------------------
# utility for testing
def main(fid,pp):

	fr = fidReader(fid)
	fr.print_header()
	data, hdr = fr.read()
	
	rec = recon(pp,data)
	kspace, imspace = rec.recon()
	#rec.recon()
	w_im = niftiWriter(pp,imspace)
	w_k = niftiWriter(pp,kspace)
	w_k.write_from_fid('out_kspace')
	w_im.write_from_fid('out_imspace')
if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('fid',help='input .fid file')
	args = parser.parse_args()
	if os.path.isdir(args.fid):
		fid = str(args.fid)+'/fid'
		pp = str(args.fid)+'/procpar'
	else:
		fid = args.fid
	main(fid,pp)
