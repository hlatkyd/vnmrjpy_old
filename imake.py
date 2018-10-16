#!/usr/bin/python3


import sys
import numpy as np
sys.path.append('/home/david/dev/vnmrjpy')
from kmake import kSpaceMaker
from argparse import ArgumentParser
from readfid import fidReader
from readprocpar import procparReader
from writenifti import niftiWriter
from niplot import niPlotter
import matplotlib.pyplot as plt
import nibabel as nib

class imgSpaceMaker():
    """
    Class to reconstruct MR images to real space from k-space.
    Generally this is done by fourier transform and corrections.
    Various compressed sensing approaches should be added here

    Hardcoded for each 'seqfil' sequence
    """

    def __init__(self, kspace, procpar, skiptab=None, skipint=None):
        """
        kspace = np.ndarray([receivers, phase, read, slice, echo])
        procpar = /procpar/file/path
        For CS reconstruction:
        skiptab = 'y' or 'n' procpar parameter
        skipint = procpar parameter to show which k space lines are excluded
        """
        self.skipint = skipint
        self.skiptab = skiptab
        self.kspace_data = kspace
        self.p = procparReader(procpar).read()
        
    def make(self):
        """
        Fourier reconstruction

        # TODO: dc corr, noise corr etc...

        return imgspace = numpy.ndarray()

        """
        seqfil = str(self.p['seqfil'])
        if seqfil in ['gems', 'fsems', 'mems', 'sems', 'mgems']:

            #print(seqfil)
            img_space = np.fft.ifft2(self.kspace_data, axes=(1,2), norm='ortho')
            img_space = np.fft.fftshift(img_space, axes=(1,2))
            #print('imgspacemaker: kspace data: '+str(self.kspace_data.shape))
            #print('imgspacemaker: imgspace shape :' +str(img_space.shape))
        elif seqfil in ['ge3d','fsems3d']:
            
            img_space = np.fft.ifftn(self.kspace_data, axes=(1,2,3), norm='ortho')
            img_space = np.fft.fftshift(img_space, axes=(1,2,3))

        elif seqfil in ['ge3d_elliptical']:

            img_space = np.fft.ifftn(self.kspace_data, axes=(1,2,3), norm='ortho')
            img_space = np.fft.fftshift(img_space, axes=(1,2,3))

        else:
            raise Exception('Sequence reconstruction not implemented yet')
   
        return img_space

    def make_cs_image_sparse(self):
        """
        Do Compressed Sensing reconstruction if image is sparse in
        image space.
        """

        pass

    def make_cs_wavelet_sparse(self):
        """
        Do this reconstruction if image is sparse in wavelet space
    
        """
        pass

if __name__ == '__main__':

    #parser = ArgumentParser()
    #parser.add_argument('inputfile')

    #args = parser.parse_args()

    # test data is kmake_test.nii.gz
    TESTFILE_REAL = 'tempdata/kmake_test_real.nii.gz'
    TESTFILE_IMAG = 'tempdata/kmake_test_imag.nii.gz'
   
    # TODO
    # for testing add another dim to kspace data to axis0
     
    TESTPP = 'tempdata/kmake_test.procpar'

    TEST_OUTPUT = 'tempdata/imake_test'

    kspace_real = nib.load(TESTFILE_REAL).get_fdata()
    kspace_imag = nib.load(TESTFILE_IMAG).get_fdata()
    
    kspace = np.vectorize(complex)(kspace_real, kspace_imag)

    imgspacemaker = imgSpaceMaker(kspace, TESTPP)
    img = imgspacemaker.make()
    print('img shape: '+str(img.shape))
    img_to_write = np.absolute(img[...,0])

    niPlotter(img_to_write).plot_slices()

    writer = niftiWriter(TESTPP, img_to_write)
    writer.write(TEST_OUTPUT)


    
