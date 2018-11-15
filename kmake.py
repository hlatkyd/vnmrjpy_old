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

    Leave rest of reconstruction to other classes/functions

    INPUT:  fid data = np.ndarra([blocks, np*traces])
            procpar
            fid header

    METHODS:

            make():


            return kspace = nump.ndarray([rcvrs, phase, read, slice, echo*time])

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
            Makes phase encode skip table based on 32 bit skipint
            will be Used for Compressed sensing sequences and ge3D_elliptical
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

            elif p['seqfil'] == 'epip':  # or other

                pass
                #TODO

            else:
                print('kmake: skipint table not implemented for this sequence')
                raise(Exception)
        
        def build_reduced_kspace(pre_data, skip_matrix, final_shape):
            """
            Arrange sparse readout lines in proper order in full kspace
            will be Used for Compressed sensing and ge3D_elliptical
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

        def make_init_shape(p):
            """
            Determines final shape of kspace based on procpar
            input: ppdict (procpar dixctionary made by procparReader) 
            """
            
            rcvrs = str(p['rcvrs']).count('y')  # count the number of y-s. ex.: rcvrs='yynn'             

            if p['apptype'] in ['im2D', 'im2Dfse']:
                read = int(p['np'])//2
                phase = int(p['nv'])
                slices = int(p['ns'])
                try:
                    echo = int(p['ne'])  # number of echoes in multiecho seq
                except:
                    echo = 1
                #TODO  important, especially for bssfp
                #time = int(p['images'])  # it can be made by arraying TR
                                        # should account for this somehow
                time = 1

                return (rcvrs, phase, slices, echo*time, read)

            elif p['apptype'] in ['im2Depi']:

                read = int(p['nread'])//2
                phase = int(p['nphase'])
                slices = int(p['ns'])
                echo = 1
                time = int(p['images'])

                return (rcvrs, phase, slices, echo*time, read)

            elif p['apptype'] in ['im3D', 'im3Dfse']:

                read = int(p['np'])//2
                phase = int(p['nv'])
                phase2 = int(p['nv2'])
                try:
                    echo = int(p['ne'])  # number of echoes in multiecho seq
                except:
                    echo = 1
                time = 1

                return (rcvrs, phase, phase2, echo*time, read)

        def make_preshape_and_shape(seqcon, shape_init):
            """
            determines preshape matrix
            seqcon = 'nccsn' = compressed/segmented(?)/none
                characters mean: [echoes,slices,phase1,pahse2,phase3]

            """
            #TODO this is disgusting... gotta rethink sometime...

            seqcon = str(seqcon)  # just to be sure...
            shape_init = list(shape_init)
            if seqcon.count('s') == 0:

                swapindex = [0,1,2,3,4]

                preshape = (shape_init[0],np.prod(shape_init[1:]))

                return (preshape, shape_init, swapindex)

            elif seqcon.count('s') == 1:

                swapindex = [0,1,2,3,4]
                pos = seqcon.index('s')
                ind_list = [3,2,1,2,4]  # according to seqcon, see spinsights for explanation....
                shape_init_temp = shape_init.copy()
                swapindex_temp = swapindex.copy()
                print('shape_init before del in make ps.. : {}'.format(shape_init))
                del shape_init_temp[0]
                del swapindex_temp[0]
                del shape_init_temp[ind_list[pos]-1]
                del swapindex_temp[ind_list[pos]-1]

                print('shape_init_temp: {}'.format(shape_init_temp))
                preshape = (shape_init[0], shape_init[ind_list[pos]], np.prod(shape_init_temp))
                print('Preshape in make ps.. : {}'.format(preshape))
                print('shape_init in make ps.. : {}'.format(shape_init))

                shape = tuple([shape_init[0], shape_init[ind_list[pos]]] + shape_init_temp)
                swapindex = [0,ind_list[pos]] + swapindex_temp

                return (preshape, shape, swapindex)
        
            else:

                raise(Exception('more than 1 "s" in seqcon not implemented yet. Returning...'))


        def standard_fill_kspace(p, shape, preshape, swapindex):

            if p['apptype'] in ['im2D', 'im3D']:
                """
                if shape == preshape:

                    kspace = np.vectorize(complex)(self.data[:,0::2], self.data[:,1::2])
                    kspace = np.reshape(kspace, shape, order='C')
                    kspace = np.moveaxis(kspace, [0,4,1,2,3], [0,1,2,3,4])
                else:
                """
                kspace = np.vectorize(complex)(self.data[:,0::2], self.data[:,1::2])
                kspace = np.reshape(kspace, preshape, order='F')
                kspace = np.reshape(kspace, shape, order='C')
                kspace = np.moveaxis(kspace, swapindex, [0,1,2,3,4])
                kspace = np.moveaxis(kspace, [0,4,1,2,3], [0,1,2,3,4])
        
                if int(p['sliceorder']) == 1:  # sliceorder = 1 means interleved slices
                    c = np.zeros(kspace.shape, dtype=complex)
                    c[:,:,:,1::2,:] = kspace[:,:,:,kspace.shape[3]//2:,:]
                    c[:,:,:,0::2,:] = kspace[:,:,:,:kspace.shape[3]//2,:]
                    kspace = c

                return kspace

            elif p['apptype'] in ['im2Depi']:

                pass

            elif p['apptype'] in ['im2Dfse']:

                pass
    
            else:
                print('apptype not implemented')

        #--------------------------MAKING K-SPACE------------------------------

        p = self.p # for ease of use...
       
        init_shape = make_init_shape(p)

        (preshape, shape, swapindex) = make_preshape_and_shape(p['seqcon'], init_shape)
 
        if 'skiptab' in p:
            if p['skiptab'] == 'y':
                # k space lines may be zeros if there is skiptab param in procpar
                skip_matrix = decode_skipint(p['skipint'])
                kspace = standard_fill_kspace(p, shape, preshape, swapindex)
                kspace = build_reduced_kspace(pre_kspace, skip_matrix, shape)
            else:
                # standard recon
                kspace = standard_fill_kspace(shape, preshape, swapindex)

        else:
            # standard recon
            kspace = standard_fill_kspace(p, shape, preshape, swapindex)

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

