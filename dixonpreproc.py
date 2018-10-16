#!/usr/bin/python3

"""TODO: correcting gaps in fieldmap;  Write preproc_data file with more content etc fieldmap name"""

import os
import sys
from shutil import copyfile
sys.path.append('/home/david/bin')
sys.path.append('/home/david/dev/common')
from readprocpar import procparReader
from readfdf import fdfReader
from writenifti import niftiWriter

import matplotlib.pyplot as plt
import numpy as np
import glob
import math
import time

class dixonPreproc():
    """
    Description:
        Makes a directory 'proc_dixon' in parent 'study_dir'. Reads rawRE.img, rawIM.img
        kspace data (.fdf files in .img directory), does inverse fourier transform. and
        converts to Nifti1. The Niftis are saved in 'proc_dixon' directory. FSL prelude
        is called to do phase unwrapping on with the Niftis as input. Unwrapped phase is
        also saved as Nifti.
    """
    def __init__(self, study_dir, pslabel='fsems2', make_fieldmap=True, unwrap_all=False):

        super(dixonPreproc, self).__init__()
        self.study_dir = study_dir
        self.proc_dir = self.study_dir+'/proc_dixon'
        self.preproc_data = self.study_dir+'/proc_dixon/preproc_data.txt'
        self.make_fieldmap = make_fieldmap
        self.unwrap_all = unwrap_all
        self.pslabel = pslabel
        image_list = glob.glob(study_dir+'/'+pslabel+'*img')
        if not image_list:
            print(str(pslabel)+' not found in '+str(study_dir))
            return
        rawre = sorted([i for i in image_list if 'rawRE' in i ])
        rawim = sorted([i for i in image_list if 'rawIM' in i ])
        ind = [i for i in range(len(rawre))] # all indices to use, might be less for less-point-dixon
        self.combined_names = [[rawre[i],rawim[i]] for i in ind]

        if not rawre: # if empty, should preproc from .fid files
            self.preproc_from_fid = True
        else:
            self.preproc_from_fid = False

    def run(self):

        def make_fieldmap(minus_pi_data, plus_pi_data, procpar):
            """
            Make phase correction factor from minus Pi shifted and plus Pi shifted data. Does 
            unwrapping with prelude in two steps: first step is preliminary field map with output
            mask, then this output mask is holefilled and the unwrapping is done again
            """
            output_name_fieldmap = self.proc_dir+'/fieldmap_wrapped'
            output_name_fieldmap_uw = self.proc_dir+'/fieldmap_unwrapped'
            output_name_fieldmap_mask = self.proc_dir+'/fieldmap_premask'
            output_name_fieldmap_mask_corr = self.proc_dir+'/fieldmap_mask'
            magnitude_name_for_prelude_corr = self.proc_dir+'/masked_abs'

            data = np.multiply(plus_pi_data,np.conj(minus_pi_data))
            fieldmap = np.arctan2(np.imag(data),np.real(data))
            niftiwriter = niftiWriter(procpar, fieldmap)
            niftiwriter.write(output_name_fieldmap)

            print('Unwrapping fieldmap ...')
            # make initial field map 
            os.system('prelude -p '+output_name_fieldmap+' -a '+magnitude_name_for_prelude+ \
                        ' -u '+output_name_fieldmap_uw+' --savemask='+output_name_fieldmap_mask)
            # close gaps in mask
            os.system('fslmaths '+output_name_fieldmap_mask+' -fillh '+output_name_fieldmap_mask_corr)
            # make new absolute with mask
            os.system('fslmaths '+magnitude_name_for_prelude+' -mul '+output_name_fieldmap_mask_corr+\
                        ' '+magnitude_name_for_prelude_corr)
            # make final field map
            os.system('prelude -p '+output_name_fieldmap+' -a '+magnitude_name_for_prelude_corr+ \
                        ' -u '+output_name_fieldmap_uw)
            print('Fieldmap done!')
            return
            
        def inverse_fourier_transform2D(kspace_data):
            
            kspace_data = np.fft.fftshift(kspace_data,axes=(2,3))
            imgspace_data = np.fft.ifft2(kspace_data,axes=(2,3), norm='ortho')
            imgspace_data = np.fft.ifftshift(imgspace_data, axes=(2,3))
            return imgspace_data

        if not os.path.exists(self.proc_dir):
            os.makedirs(self.proc_dir)

        if not self.preproc_from_fid:

            open(self.preproc_data, 'w').close()

            procpar = []
            roshift = []
            for item in self.combined_names:
                # combined_names is a list of lists containing real and imaginary parts    
                procpar.append(item[0]+'/procpar')
                ppr = procparReader(item[0]+'/procpar')
                shift = int(float(ppr.read()['roshift'])*1000000)  # write it in microsec for naming
                roshift.append(float(ppr.read()['roshift']))
                hdr , data_re = fdfReader(item[0],'out').read()
                hdr , data_im = fdfReader(item[1],'out').read()
                kspace_data = np.vectorize(complex)(data_re[0,...],data_im[0,...])
                imgspace_data = inverse_fourier_transform2D(kspace_data)
                magnitude_data = np.absolute(imgspace_data)
                phase_data = np.arctan2(np.imag(imgspace_data),np.real(imgspace_data))
                
                niftiwriter = niftiWriter(item[0]+'/procpar',magnitude_data)
                output_name_magnitude = self.proc_dir+'/'+self.pslabel+'_'+str(shift)+'us_mag'
                niftiwriter.write(output_name_magnitude)
                niftiwriter = niftiWriter(item[1]+'/procpar',phase_data)
                output_name_phase = self.proc_dir+'/'+self.pslabel+'_'+str(shift)+'us_ph'
                niftiwriter.write(output_name_phase)
                output_name_unwrapped = self.proc_dir+'/'+self.pslabel+'_'+str(shift)+'us_unwrapped_ph'

                if float(ppr.read()['roshift']) == 0.00037:
                    plus_pi_data = imgspace_data
                if float(ppr.read()['roshift']) == -0.00037:
                    minus_pi_data = imgspace_data
                if float(ppr.read()['roshift']) == 0:
                    magnitude_name_for_prelude = output_name_magnitude
                    copyfile(item[0]+'/procpar', self.proc_dir+'/procpar')
            
                if self.unwrap_all:
                    print('Unwrapping phasemaps ...')
                    os.system('prelude -p '+output_name_phase+' -a '+output_name_magnitude+ \
                                ' -u '+output_name_unwrapped)
                with open(self.preproc_data,'a') as openfile:
                    line = str(float(ppr.read()['roshift']))+','\
                            +output_name_magnitude+','\
                            +output_name_phase+'\n'
                    openfile.write(line)                

        if self.preproc_from_fid:
    
            pass


        if self.make_fieldmap:

            make_fieldmap(minus_pi_data, plus_pi_data, procpar[0])

        return
        
if __name__ == '__main__':
        
    study_dir = '/home/david/dev/dixon/s_2018080901'
    pslabel = 'fsems2'
    dixpp = dixonPreproc(study_dir, pslabel)
    dixpp.run()
    #dixonPreproc(study_dir,pslabel).run()



