#!/usr/local/bin/python3.6

import sys
import os
sys.path.append('/home/david/work/vnmrjpy')
sys.path.append('/home/david/dev/vnmrjpy')
import numpy as np
import nibabel as nib
from argparse import ArgumentParser
from readfid import fidReader
from readprocpar import procparReader
from writenifti import niftiWriter
from niplot import niPlotter
import matplotlib.pyplot as plt



#WARNING!!! WARNING !!!! CLASS IS HARDCODED FOR EACH seqfil AND seqcon !!!!



class kSpaceMaker():
    """
    Class to build the k-space from the raw fid data based on procpar.
    raw fid_data is numpy.ndarray(blocks, traces * np) format. Should be
    untangle based on 'seqcon' or 'seqfil' parameters.

    Should support compressed sensing

    Leave rest of reconstruction to other classes/functions
    """
    def __init__(self, fid_data, procpar, fidheader):

        self.data = fid_data
        self.p = procparReader(procpar).read()
        self.fhdr = fidheader
        print('Making k-space for '+str(self.p['seqfil'])+' seqcon: '+str(self.p['seqcon']))

    def print_fid_header(self):
        for item in self.fhdr.keys():
            print(str('{} : {}').format(item, self.fhdr[item]))

    def make(self):
        """
        Main method.
        Make separate section for each option to fine tune.

        Return kspace

        """
        #---------------------------HELPER FUNCTIOINS-----------------------------------
        def decode_skipint(skipint):
            """
            Makes phase encode skit table based on 32 bit skipint
            """
            p = self.p
            BITS = 32
            print('skipint len: '+str(len(skipint)))
            if p['seqfil'] in ['ge3d', 'ge3d_elliptical', 'mprage3d']:
            
                skip_matrix = np.zeros([int(p['nv']), int(p['nv2'])])

                skipint = [int(x) for x in skipint]
                skipint_binary_vals = [str(np.binary_repr(d, BITS)) for d in skipint]
                skipint_binary_vals = ''.join(skipint_binary_vals)
                skipint_binary_array = np.asarray([int(i) for i\
                                                 in skipint_binary_vals])

                print('nv, nv2, np: '+str(p['nv'])+' '+str(p['nv2'])+' '+str(p['np']))

                skip_matrix = np.reshape(skipint_binary_array, skip_matrix.shape)
                #plt.imshow(skip_matrix)
                #plt.show()
                return skip_matrix

            elif p['seqfil'] == 'fsems':  # or other

                pass
                #TODO

            else:
                print('kmake: skipint table not implemented for this sequence')
                raise(Exception)
        
        def build_kspace(pre_data, skip_matrix, final_shape):
            """
            Arrange sparse readout lines in proper order in full kspace
            """
            p = self.p
            kspace = np.zeros(final_shape, dtype=complex)
            if p['seqfil'] in ['ge3d', 'ge3d_elliptical', 'mprage3d']:

                if p['seqcon'] == 'ncccn':

                    # input: [rcvrs, pre_phase, echo, read]
                    # output: [rcvrs, phase, phase2, echo, read]
                    n = int(p['nv'])
                    count = 0
                    for i in range(skip_matrix.shape[0]):
                        for k in range(skip_matrix.shape[1]):
                            if skip_matrix[i,k] == 1:
                                kspace[:,i,k,:,:] = pre_data[:,count,:,:]
                                count = count+1

            return kspace

        #--------------------------MAKING K-SPACE------------------------------

        p = self.p # for ease of use...
        rcvrs = 4 if p['rcvrs'] == 'yyyy' else 1  # number of receiver channels, now only 4 or 1
        
        if p['seqfil'] == 'gems':

            if p['seqcon'] == 'nccnn':
                
                phase = int(p['nv'])
                read = int(p['np'])//2
                slices = int(p['ns'])
                echo = 1
                shape = (rcvrs, phase, slices, echo, read) # time dim 1
                kspace = np.vectorize(complex)(self.data[:,0::2], self.data[:,1::2])
                kspace = np.reshape(kspace, shape, order='c')
                kspace = np.swapaxes(kspace.swapaxes(2, 4), 3, 4)
    
                if int(p['sliceorder']) == 1:  # sliceorder = 1 means interleved slices
                    c = np.zeros(kspace.shape, dtype=complex)
                    c[:,:,:,1::2,:] = kspace[:,:,:,slices//2:,:]
                    c[:,:,:,0::2,:] = kspace[:,:,:,:slices//2,:]
                    kspace = c

            return kspace
                    
                    
        if p['seqfil'] == 'sems':

            if p['seqcon'] == 'ncsnn':
                
                phase = int(p['nv'])
                read = int(p['np'])//2
                slices = int(p['ns'])
                echo = 1
                preshape = (rcvrs, phase, slices*echo*read)
                shape = (rcvrs, phase, slices, echo, read) # time dim 1
                kspace = np.vectorize(complex)(self.data[:,0::2], self.data[:,1::2])
                kspace = np.reshape(kspace, preshape, order='F')
                kspace = np.reshape(kspace, shape, order='C')
                kspace = np.swapaxes(kspace.swapaxes(2, 4), 3, 4)
    
                if int(p['sliceorder']) == 1:  # sliceorder = 1 means interleved slices
                    c = np.zeros(kspace.shape, dtype=complex)
                    print(c.shape)
                    print(kspace.shape)
                    c[:,:,:,1::2,:] = kspace[:,:,:,slices//2:,:]
                    c[:,:,:,0::2,:] = kspace[:,:,:,:slices//2,:]
                    kspace = c

            return kspace

        if p['seqfil'] == 'fsems':

            pass

        if p['seqfil'] == 'epip':

            pass

        if p['seqfil'] == 'ge3d':

            # TODO

            if p['seqcon'] == 'nccsn':

                phase = int(p['nv'])
                read = int(p['np'])//2
                phase2 = int(p['nv2'])
                echo = 1
                shape = (rcvrs, phase2, phase, echo, read) # time dim 1
                preshape = (rcvrs, phase2, phase*echo*read)
                
                kspace = np.vectorize(complex)(self.data[:,0::2], self.data[:,1::2])
                kspace = np.reshape(kspace, preshape, order='F')
                kspace = np.reshape(kspace, shape, order='c')
                kspace = np.swapaxes(kspace.swapaxes(1, 4), 3, 4)

            return kspace

        #-------------------------------------------------------------------------
        #                       COMPRESSED SENSING K-SPACE
        #------------------------------------------------------------------------

        if p['seqfil'] == 'ge3d_elliptical': # test sequence

            if p['seqcon'] == 'ncccn' and p['skiptab'] == 'y':

                skip_matrix = decode_skipint(p['skipint'])
                phase = int(p['nv'])
                phase2 = int(p['nv2'])
                read = int(p['np'])//2
                #slices = int(p['ns'])
                pre_phase = int(self.fhdr['ntraces'])
                echo = 1
                shape = (rcvrs, phase, phase2, echo, read) # time dim 1
                pre_shape = (rcvrs, pre_phase, echo, read)
                pre_kspace = np.vectorize(complex)(self.data[:,0::2], self.data[:,1::2])
                pre_kspace = np.reshape(pre_kspace, pre_shape, order='c')
                
                kspace = build_kspace(pre_kspace, skip_matrix, shape)
                kspace = np.swapaxes(kspace.swapaxes(2, 4), 3, 4)
    
            return kspace


if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('input_fid_dir')  # just give the whole directory
    args = parser.parse_args()
    fid = args.input_fid_dir + '/fid'
    procpar = args.input_fid_dir + '/procpar'
    fidrdr = fidReader(fid, procpar)
    data, fidheader = fidrdr.read()
    fidrdr.print_header()
    kspacemaker = kSpaceMaker(data, procpar, fidheader)
    #kspacemaker.print_fid_header()
    kspace = kspacemaker.make()


    kspace_real = np.real(kspace[0,...])
    kspace_imag = np.imag(kspace[0,...])

    writer_re = niftiWriter(procpar, kspace_real)
    writer_re.write('tempdata/kmake_test_real')
    writer_im = niftiWriter(procpar, kspace_imag)
    writer_im.write('tempdata/kmake_test_imag')

    plotter = niPlotter(kspace_real)
    plotter.plot_slices()

    os.system('cp '+str(procpar)+' tempdata/kmake_test.procpar')

