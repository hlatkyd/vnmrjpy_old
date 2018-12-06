#!/usr/bin/python3

import os
import sys
sys.path.append('/home/david/bin')
sys.path.append('/home/david/dev/common')
from readprocpar import procparReader
from readfdf import fdfReader
from writenifti import niftiWriter

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import glob
import math
import time

class dixonProc():
    """    
    Class for fat fraction calculation with dixon method.  
    """
    def __init__(self,\
                study_dir,\
                points=None,\
                fieldmap_corr=True,\
                check_preproc=True,\
                freq=[0,1400],\
                saving=True):
        self.study_dir = study_dir
        self.points = points
        self.check_preproc = check_preproc
        self.freq = freq
        self.proc_dir = study_dir+'/proc_dixon'
        self.base_procpar = study_dir+'/proc_dixon/procpar'
        self.fieldmap= study_dir+'/proc_dixon/fieldmap_unwrapped.nii.gz' # 'fieldmap' in radians
        self.preproc_data = study_dir+'/proc_dixon/preproc_data.txt'
        self.saving = saving

        if self.check_preproc:
            self.proc_check_ok = os.path.isdir(self.proc_dir) \
                            and os.path.isfile(self.base_procpar) \
                            and os.path.isfile(self.fieldmap)
        
    def save_results_to_nifti(self, out, affine, header):

        # out shape is [ro1re, ro1im, ro2re, ro2im, y1re, y1im ....] (9,read,phase,slice,1)

        ro = out[0:2*len(self.freq),...]
        y = out[2*len(self.freq):-1,...]

        roshape = tuple([len(self.freq)])+out[0].shape
        ro = np.ones(roshape,dtype=complex)
        yshape = tuple([1+len(self.freq)])+out[0].shape
        y = np.ones(yshape,dtype=complex)
        for i in range(roshape[0]):
            ro[i,...] = np.vectorize(complex)(out[i*2,...], out[i*2+1,...])

        for i in range(1,yshape[0]): # leaving out delta_field
            y[i,...] = np.vectorize(complex)(out[len(self.freq)*2-1+i*2,...], \
                                                out[len(self.freq)*2+i*2,...])        

        if len(self.freq) == 2: # hardcode, because probably wont be used for other than 2

            # save fat and water fractions
            water = np.abs(ro[0,...])
            fat = np.abs(ro[1,...])
            water_nifti = nib.Nifti1Image(np.divide(water,water+fat), affine)
            fat_nifti = nib.Nifti1Image(np.divide(fat,water+fat), affine)
            water_out = self.proc_dir+'/water.nii.gz'
            fat_out = self.proc_dir+'/fat.nii.gz'
            mask = self.proc_dir+'/fieldmap_mask.nii.gz'
            nib.save(water_nifti, water_out)
            nib.save(fat_nifti, fat_out)
            os.system('fslmaths '+water_out+' -mul '+mask+' '+water_out)
            os.system('fslmaths '+fat_out+' -mul '+mask+' '+fat_out)

            # save fat and water fraction errors

            water_err = np.abs(y[1,...])
            fat_err = np.abs(y[2,...])
            water_err_nifti = nib.Nifti1Image(np.divide(water_err, water), affine)
            fat_err_nifti = nib.Nifti1Image(np.divide(fat_err, fat), affine)
            water_err_out = self.proc_dir+'/water_err.nii.gz'
            fat_err_out = self.proc_dir+'/fat_err.nii.gz'

            nib.save(water_err_nifti, water_err_out)
            nib.save(fat_err_nifti, fat_err_out)
            os.system('fslmaths '+water_err_out+' -mul '+mask+' '+water_err_out)
            os.system('fslmaths '+fat_err_out+' -mul '+mask+' '+fat_err_out)

        else:
            print('Oh please, if there are other than 2 dixon components, \
                    you have to finish the code ... So... no results yet...')


    def least_squares_fit(self):
        """
        Non-iterative least squares fitting

        """
        def read_proc_dir():

            roshift = []
            data = []

            with open(self.preproc_data,'r') as openfile:
                lines = [line.rstrip('\n') for line in openfile]
            for line in lines:
                roshift.append(float(line.split(',')[0]))
                mag_nifti = line.split(',')[1]+'.nii'
                ph_nifti = line.split(',')[2]+'.nii'
                mag = nib.load(mag_nifti)
                ph = nib.load(ph_nifti)
                nifti_hdr = mag.header
                nifti_affine = mag.affine
                imgspace_data = mag.get_fdata() * np.exp(1j*ph.get_fdata())
                data.append(imgspace_data)

            field_nii = nib.load(self.fieldmap)
            field = np.expand_dims(field_nii.get_fdata(),axis=-1) # make 4d to fit dixon data

            return np.asarray(data), field, roshift, nifti_hdr, nifti_affine

        def make_Acd(roshift, freq):
            A = np.zeros((2*len(roshift),2*len(freq)))
            c = np.asarray([math.cos(2*np.pi*i*j) for i in freq for j in roshift])
            d = np.asarray([math.sin(2*np.pi*i*j) for i in freq for j in roshift])
            c = np.reshape(c,(len(freq),len(roshift)),order='c')
            d = np.reshape(d,(len(freq),len(roshift)),order='c')
            
            A[:len(roshift),0::2] = c.T
            A[len(roshift):,0::2] = d.T
            A[:len(roshift),1::2] = -d.T
            A[len(roshift):,1::2] = c.T 

            return A, c, d        

        def apply_field_correction(data, field, roshift):

            for k, shift in enumerate(roshift):
                data[k,...] = data[k,...]*np.exp(-1j*2*np.pi*field*shift)            
            return data

        def apply_T2_correction(data, t2map, roshift):

            for k, shift in enumerate(roshift):
                data[k,...] = data[k,...]*np.exp(-1j*2*np.pi*field*shift)            
            return data

        def fit_voxelvise(data1D, roshift, A, c, d, freq):
            """
            Voxelvise least-squares fitting to dixon data. Returns the fit result (ro)
            and the error terms y (delta_field, delta_ro1, etc..) 
            """
            def S_complex_to_real(data1D, field, freq, roshift):

                S = data1D
                Sre = np.asarray([np.real(S[i]) for i in range(S.shape[0])])
                Sim = np.asarray([np.imag(S[i]) for i in range(S.shape[0])])

                return np.concatenate((Sre, Sim),axis=0).T # return col vector
            
            def ro_from_S(S,A):

                return np.linalg.multi_dot((np.linalg.inv(np.dot(A.T,A)),A.T,S))

            def B_from_ro(ro, A, c, d, roshift, freq):

                B = np.zeros((A.shape[0],A.shape[1]+1))
                B[:,1:] = A
                rore = ro[0::2]
                roim = ro[1::2]
                gre = [2*np.pi*roshift[i]*(-rore[j]*d[j,i]-roim[j]*c[j,i]) \
                         for i in range(len(roshift)) for j in range(len(freq))] # g1N_re
                gre = np.sum(np.reshape(gre,(c.shape),order='F'),axis=0)

                gim = [2*np.pi*roshift[i]*(rore[j]*c[j,i]-roim[j]*d[j,i]) \
                         for i in range(len(roshift)) for j in range(len(freq))] # g1N_re
                gim = np.sum(np.reshape(gim,(c.shape),order='F'),axis=0)
            
                B[:len(gre),0] = gre
                B[len(gim):,0] = gim

                return B

            def y_from_S(S,B):# y contains the error terms delta_field, delta_ro     

                return np.linalg.multi_dot((np.linalg.inv(np.dot(B.T,B)),B.T,S))

            #---------------------------------------------------------------------------------------

            #---------------------------------------------------------------------------------------

            S = S_complex_to_real(data1D, field, freq, roshift)
            ro = ro_from_S(S, A)
            B = B_from_ro(ro, A, c, d, roshift, self.freq)
            y = y_from_S(S, B)

            return np.concatenate([ro, y])

        
        """-------------------------least_squares_fit MAIN-------------------------------------"""

        if self.proc_check_ok:
            print('dixonProc.least_squares_fit() : Preprocess check OK')
        else:
            print('dixonProc.least_squares_fit() : Preprocess was not done. \
                    Use dixonPreproc().run() first')
            return
        # data.shape : [roshift, read, phase, slice, time=1]
        # field.shape : [read, phase, slice, 1]

        print('Calculating water and fat ratio by least-squares fitting ... ')
        data, field, roshift, header, affine = read_proc_dir()
        data = apply_field_correction(data, field, roshift)
        #data = apply_T2_correction(data, t2map, roshift)
        A, c, d = make_Acd(roshift, self.freq)
        out = np.apply_along_axis(fit_voxelvise, 0, data, roshift, A, c, d, self.freq)

        if self.saving:
            self.save_results_to_nifti(out, affine, header)
        print('Done!')        

        return out

    def ideal_fit(self, iterations, masked=False, fieldcorr=False):
        """
        IDEAL method
        """
        def read_proc_dir():

            roshift = []
            data = []

            with open(self.preproc_data,'r') as openfile:
                lines = [line.rstrip('\n') for line in openfile]
            for line in lines:
                roshift.append(float(line.split(',')[0]))
                try:
                    mag_nifti = line.split(',')[1]+'.nii.gz'
                    ph_nifti = line.split(',')[2]+'.nii.gz'
                except:
                    mag_nifti = line.split(',')[1]+'.nii'
                    ph_nifti = line.split(',')[2]+'.nii'
                mag = nib.load(mag_nifti)
                ph = nib.load(ph_nifti)
                nifti_hdr = mag.header
                nifti_affine = mag.affine
                imgspace_data = mag.get_fdata() * np.exp(1j*ph.get_fdata())
                data.append(imgspace_data)

            field_nii = nib.load(self.fieldmap)
            field = np.expand_dims(field_nii.get_fdata(),axis=-1) # make 4d to fit dixon data

            return np.asarray(data), field, roshift, nifti_hdr, nifti_affine

        def apply_field_correction(data, field, roshift):

            for k, shift in enumerate(roshift):
                data[k,...] = data[k,...]*np.exp(-1j*2*np.pi*field*shift)            
            return data

        def recalc_ro_final(ro, A, field, S):
        
            fullA = np.repeat(A,S.size).reshape(A.shape + S.shape,order='c')
            ro =  np.linalg.multi_dot((np.linalg.inv(np.dot(A.T,A)),A.T,S))
            return ro

        def make_Acd(roshift, freq):
            A = np.zeros((2*len(roshift),2*len(freq)))
            c = np.asarray([math.cos(2*np.pi*i*j) for i in freq for j in roshift])
            d = np.asarray([math.sin(2*np.pi*i*j) for i in freq for j in roshift])
            c = np.reshape(c,(len(freq),len(roshift)),order='c')
            d = np.reshape(d,(len(freq),len(roshift)),order='c')
            
            A[:len(roshift),0::2] = c.T
            A[len(roshift):,0::2] = d.T
            A[:len(roshift),1::2] = -d.T
            A[len(roshift):,1::2] = c.T 

            return A, c, d

        def fit_voxelvise(data1D, roshift, A, c, d, F0=0, freq=[0,1400]):
            """
            1d function to be used for apply_along_axis
            F0 is initial field
            freq is the Dixon components frequency shift
            """
            def Sh_from_f(data1D, field, freq, roshift):

                Sh = data1D*np.exp(-1j*2*np.pi*field*np.asarray(roshift))
                Sre = np.asarray([np.real(Sh[i]) for i in range(Sh.shape[0])])
                Sim = np.asarray([np.imag(Sh[i]) for i in range(Sh.shape[0])])

                return np.concatenate((Sre, Sim),axis=0).T # return col vector
            
            def ro_from_Sh(S,A):

                return np.linalg.multi_dot((np.linalg.inv(np.dot(A.T,A)),A.T,S))

            def B_from_ro(ro, A, c, d, roshift, freq):

                B = np.zeros((A.shape[0],A.shape[1]+1))
                B[:,1:] = A
                rore = ro[0::2]
                roim = ro[1::2]
                gre = [2*np.pi*roshift[i]*(-rore[j]*d[j,i]-roim[j]*c[j,i]) \
                         for i in range(len(roshift)) for j in range(len(freq))] # g1N_re
                gre = np.sum(np.reshape(gre,(c.shape),order='F'),axis=0)
                #gre = np.reshape(gre,(c.shape),order='F')
                gim = [2*np.pi*roshift[i]*(rore[j]*c[j,i]-roim[j]*d[j,i]) \
                         for i in range(len(roshift)) for j in range(len(freq))] # g1N_re
                gim = np.sum(np.reshape(gim,(c.shape),order='F'),axis=0)
                #gre = np.reshape(gre,(c.shape),order='F')
            
                B[:len(gre),0] = gre
                B[len(gim):,0] = gim

                return B

            def y_from_Sh(S,B):# y contains the error terms delta_field, delta_ro     

                return np.linalg.multi_dot((np.linalg.inv(np.dot(B.T,B)),B.T,S))

            def Sh_from_B_y(B,y):
    
                return np.dot(B,y)

            """
            actual iteration starts here
            """
            f = 0 # init fieldmap=0
            for i in range(iterations):

                Sh = Sh_from_f(data1D,f,freq,roshift)
                ro = ro_from_Sh(Sh,A)
                B = B_from_ro(ro,A,c,d,roshift,freq)
                y = y_from_Sh(Sh,B)
                f = np.asarray(f + y[0]) # recalculate field
                # make iteration stop on voxel basis
                if abs(y[0]) < 1:
                    break

            return np.concatenate([ro,y,np.atleast_1d(f)])

        """-------------------------IDEAL_fit MAIN-------------------------------------"""

        if self.proc_check_ok:
            print('dixonProc.ideal_fit() : Preprocess check OK')
        else:
            print('dixonProc.ideal_fit() : Preprocess was not done. \
                    Use dixonPreproc().run() first')
            return
        # data.shape : [roshift, read, phase, slice, time=1]
        # field.shape : [read, phase, slice, 1]

        data, field, roshift, header, affine = read_proc_dir()

        if fieldcorr == True: 

            data = apply_field_correction(data, field, roshift)

        A,c,d = make_Acd(roshift, self.freq)

        out = np.apply_along_axis(fit_voxelvise, 0, data, roshift, A, c, d, self.freq)

        if self.saving:
            self.save_results_to_nifti(out, affine, header)
        print('Done!')

        return out # data is the masked data

if __name__ == '__main__':
    
    study_dir = '/home/david/dev/dixon/s_2018080901'

    dp = dixonProc(study_dir)
    #dp.least_squares_fit()
    dp.ideal_fit(100)
    
