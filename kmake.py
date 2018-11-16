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



class kSpaceMaker():
    """
    Class to build the k-space from the raw fid data based on procpar.
    raw fid_data is numpy.ndarray(blocks, traces * np) format. Should be
    untangle based on 'seqcon' or 'seqfil' parameters.

    Should support compressed sensing
    In case of Compressed sensing the reduced kspace is filled with zeros
    to reach the intended final shape

    Leave rest of reconstruction to other classes/functions

    INPUT:  fid data = np.ndarra([blocks, np*traces])
            procpar
            fid header

    METHODS:

            make():


            return kspace = nump.ndarray([rcvrs, phase, read, slice, echo*time])

    """
    def __init__(self, fid_data, procpar, fidheader):

        def get_arrayed_AP(p):
            """
            check for arrayed acquisition parameters in procpar
            return dictionary {par : array_length}
            """
            AP_dict = {}
            for par in ['tr', 'te', 'fa']:
            
                pass

            return AP_dict

        self.fid_data = fid_data
        self.p = procparReader(procpar).read()
        self.fid_header = fidheader
        self.rcvrs = str(self.p['rcvrs']).count('y')
        self.arrayed_AP = get_arrayed_AP(self.p)

        apptype = self.p['apptype']
       
        # decoding skipint parameter
 
        print('Making k-space for '+ str(apptype)+str(self.p['seqfil'])+\
                ' seqcon: '+str(self.p['seqcon']))

    def print_fid_header(self):
        for item in self.fhdr.keys():
            print(str('{} : {}').format(item, self.fhdr[item]))

    def make(self):
        """
        Main method. Wraps child methods which are apptype specific
        return kspace
        """
        def make_im2D():
            """
            These should provide the vnmrj standard output
            """
            p = self.p
            (read, phase, slices) = (int(p['np'])//2, \
                                            int(p['nv']), \
                                            int(p['ns']))
            if 'ne' in p.keys():
                echo = int(p['ne'])
            else:
                echo = 1

            time = 1

            if p['seqcon'] == 'nccnn':
                shape = (self.rcvrs, phase, slices, echo*time, read)
                kspace = np.vectorize(complex)(self.fid_data[:,0::2],self.fid_data[:,1::2])
                kspace = np.reshape(kspace, shape, order='C')
                kspace = np.moveaxis(kspace, [0,4,1,2,3], [0,1,2,3,4])


            elif p['seqcon'] == 'nscnn':

                pass

            elif p['seqcon'] == 'ncsnn':

                preshape = (self.rcvrs, phase, slices*echo*time*read)
                shape = (self.rcvrs, phase, slices, echo*time, read)
                kspace = np.vectorize(complex)(self.fid_data[:,0::2],self.fid_data[:,1::2])
                kspace = np.reshape(kspace, preshape, order='F')
                kspace = np.reshape(kspace, shape, order='C')
                kspace = np.moveaxis(kspace, [0,4,1,2,3], [0,1,2,3,4])

            elif p['seqcon'] == 'ccsnn':

                preshape = (self.rcvrs, phase, slices*echo*time*read)
                shape = (self.rcvrs, phase, slices, echo*time, read)
                kspace = np.vectorize(complex)(self.fid_data[:,0::2],self.fid_data[:,1::2])
                kspace = np.reshape(kspace, preshape, order='F')
                kspace = np.reshape(kspace, shape, order='C')
                kspace = np.moveaxis(kspace, [0,4,1,2,3], [0,1,2,3,4])

            if int(p['sliceorder']) == 1: # 1 if interleaved slices
                c = np.zeros(kspace.shape, dtype=complex)
                c[...,1::2,:] = kspace[...,slices//2:,:]
                c[...,0::2,:] = kspace[...,:slices//2,:]
                kspace = c

            return kspace


        def make_im2Dcs():
            """
            These (*cs) are compressed sensing variants
            """

            def decode_skipint_2D(skipint):

               pass
 
            pass
        def make_im2Depi():
            pass
        def make_im2Depics():
            pass
        def make_im2Dfse():
            pass
        def make_im2Dfsecs():
            pass
        def make_im3D():

            p = self.p
            (read, phase, phase2) = (int(p['np'])//2, \
                                    int(p['nv']), \
                                     int(p['nv2']))
            if 'ne' in p.keys():
                echo = int(p['ne'])
            else:
                echo = 1

            time = 1

            if p['seqcon'] == 'nccsn':

                pass

            if p['seqcon'] == 'ncccn':
            
                pass
    
            return kspace

        def make_im3Dcs():
            """
            3D compressed sensing
            sequences : ge3d, mge3d, se3d, etc
            """
            # ----------------------Make helper functions ---------------------

            def decode_skipint_3D(skipint):
                """
                Takes 'skipint' parameter and returns a 0-1 matrix according to it
                which tells what lines are acquired in the phase1-phase2 plane
                """
                BITS = 32  # Skipint parameter is 32 bit encoded binary, see spinsights
                skip_matrix = np.zeros([int(p['nv']), int(p['nv2'])])
                skipint = [int(x) for x in skipint]
                skipint_bin_vals = [str(np.binary_repr(d, BITS)) for d in skipint]
                skipint_bin_vals = ''.join(skipint_bin_vals)
                skipint_bin_array = np.asarray([int(i) for i in skipint_bin_vals])
                skip_matrix = np.reshape(skipint_bin_array, skip_matrix.shape)

                return skip_matrix

            def fill_kspace_3D(pre_kspace, skip_matrix, shape):
                """
                Fills up reduced kspace with zeros according to skip_matrix
                returns zerofilled kspace in the final shape
                """
                kspace = np.zeros(shape, dtype=complex)
                if self.p['seqcon'] == 'ncccn':

                    n = int(self.p['nv'])
                    count = 0
                    for i in range(skip_matrix.shape[0]):
                        for k in range(skip_matrix.shape[1]):
                            if skip_matrix[i,k] == 1:
                                kspace[:,i,k,:,:] = pre_kspace[:,count,:,:]
                                count = count+1
                return kspace

            #------------------------make start -------------------------------

            p = self.p
            (read, phase, phase2) = (int(p['np'])//2, \
                                    int(p['nv']), \
                                     int(p['nv2']))
            if 'ne' in p.keys():
                echo = int(p['ne'])
            else:
                echo = 1

            time = 1

            if p['seqcon'] == 'nccsn':

                pass

            if p['seqcon'] == 'ncccn':
            
                skip_matrix = decode_skipint_3D(p['skipint'])
                pre_phase = int(self.fid_header['ntraces'])    
                shape = (self.rcvrs, phase, phase2, echo*time, read)
                pre_shape = (self.rcvrs, pre_phase, echo*time, read)
                pre_kspace = np.vectorize(complex)\
                            (self.fid_data[:,0::2],self.fid_data[:,1::2])
                pre_kspace = np.reshape(pre_kspace, pre_shape, order='c')
                kspace = fill_kspace_3D(pre_kspace, skip_matrix, shape)
                kspace = np.moveaxis(kspace, [0,4,1,2,3],[0,1,2,3,4])
                
            return kspace

        def make_im3Dute():
            pass
        
        # ----------------Handle sequence exceptions first---------------------

        print('seqfil : {}'.format(self.p['seqfil']))

        if str(self.p['seqfil']) == 'ge3d_elliptical':
           
            kspace = make_im3Dcs()
            return kspace

        #--------------------------Handle by apptype---------------------------

        if self.p['apptype'] == 'im2D':

            kspace = make_im2D()

        elif self.p['apptype'] == 'im2Dcs':

            kspace = make_im2Dcs()

        elif self.p['apptype'] == 'im2Depi':

            kspace = make_im2Depi()

        elif self.p['apptype'] == 'im2Depics':

            kspace = make_im2Depics()

        elif self.p['apptype'] == 'im2Dfse':

            kspace = make_im2Dfse()

        elif self.p['apptype'] == 'im2Dfsecs':

            kspace = make_im2Dfsecs()

        elif self.p['apptype'] == 'im3D':

            kspace = make_im3D()

        elif self.p['apptype'] == 'im3Dcs':

            kspace = make_im3Dcs()

        elif self.p['apptype'] == 'im3Dute':

            kspace = make_im3Dute()

        else:
            raise(Exception('Could not find apptype. Maybe not implemented?'))

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

