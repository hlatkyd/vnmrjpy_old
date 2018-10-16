#!/usr/bin/python3

import os
import glob
import numpy as np
import nibabel as nib
import math
from argparse import ArgumentParser
from readprocpar import procparReader
from readfid import fidReader

class niftiWriter():
    """
    Class to write Nifti1 files from procpar and image data.
    Dimensions, orientations in the input data and procpar must match!
    
    INPUT:    procpar

            data = numpy.ndarray([phase, readout, slice, time])

    METHODS:
            write(out)

                nifti output dimensions are [phase, readout, slice, time]

                nifti affine is created from the procpar data

    """
    def __init__(self,procpar,data):
        """
        procpar is initialized, Nifti affine and header is made from procpar data at class init. 
        """
        # ----------------------------INIT HELPER FUNCTIONS-----------------------------------------
        def make_affine():
    
            def Mrot(psi,phi,theta):
                # z rotation matrix
                m_z = np.array([[math.cos(theta),-math.sin(theta),0],\
                                [math.sin(theta),math.cos(theta),0],\
                                [0,0,1]])

                m_y = np.array([[math.cos(phi),0,math.sin(phi)],\
                                [0,1,0],\
                                [-math.sin(phi),0,math.cos(phi)]])
                # x rotation matrix
                m_x = np.array([[1,0,0],\
                                [0,math.cos(psi),-math.sin(psi)],\
                                [0,math.sin(psi),math.cos(psi)]])
                matrix = np.dot(np.dot(m_x,m_y),m_z)
                return matrix

            def Mpqr(dim):
                matrix = np.identity(3)*np.transpose(dim)
                return matrix

            # rotations
            psi = float(self.ppdict['psi'])
            phi = float(self.ppdict['phi'])
            theta = float(self.ppdict['theta'])
            # translation            
            pro = float(self.ppdict['pro']) # readout offset
            ppe = float(self.ppdict['ppe'])    # phase offset
            if self.ppdict['apptype'] in ['im3D']:
                ppe2 = float(self.ppdict['ppe2'])    # phase offset
                t = np.array([pro,ppe,ppe2])
            elif self.ppdict['apptype'] in ['im2D','im2Dfse','im2Depi']:
                pss0 = float(self.ppdict['pss0']) # slice offset
                t = np.array([pro,ppe,pss0])        
            # constructing the orientation matrix M
            M_rot = Mrot(psi,phi,theta)
            M_pqr = Mpqr(dim)
            M = np.dot(M_rot,M_pqr)

            affine = np.zeros((4, 4))    
            affine[:3,:3] = M
            affine[0:3,3] = t.T
            affine[3,3] = 3 

            return affine

        def make_header(aff):

            if self.ppdict['apptype'] in ['im3D','im3Dshim']:
                header = nib.nifti1.Nifti1Header()
                header.set_data_shape(matrix)
                header.set_dim_info(phase=1,freq=0,slice=2)

            elif self.ppdict['apptype'] in ['im2D']:
                header = nib.nifti1.Nifti1Header()
                #print(header)
                header.set_data_shape(matrix)
                header.set_dim_info(phase=0,freq=1,slice=2)
                header.set_xyzt_units(xyz='mm')
                header.set_qform(aff, code='scanner')
                #print(header)
                #print('header set')

            elif self.ppdict['apptype'] in ['im2Dfse','im2Depi']:
                header = nib.nifti1.Nifti1Header()
                header.set_data_shape(matrix)
                header.set_dim_info(slice=2,phase=1,freq=0)
                header.set_xyzt_units(xyz='mm')
                if self.ppdict['apptype'] == 'im2Depi':
                    header.set_slice_duration(float(self.ppdict['slicetr']))


            

            return header


        # ----------------------------MAIN INIT-------------------------------------------

        ppr = procparReader(procpar)
        self.ppdict = ppr.read()
        if len(data.shape) == 3:
            self.data = np.expand_dims(data,axis=-1)
        elif len(data.shape) == 4:
            self.data = data
        else:
            print('datashape:'+str(data.shape))
            print('Wrong shape of input data')
            return

        if self.ppdict['apptype'] in ['im2Depi']:
            dread = float(self.ppdict['lro']) / float(self.ppdict['nread'])*2*10
            dphase = float(self.ppdict['lpe']) / float(self.ppdict['nphase'])*10
            dslice = float(self.ppdict['thk'])+float(self.ppdict['gap'])
            matrix = (int(self.ppdict['nread'])/2,int(self.ppdict['nphase']))            
            dim = np.array([dread,dphase,dslice])
        if self.ppdict['apptype'] in ['im2Dfse','im2D']:
            dread = float(self.ppdict['lro']) / float(self.ppdict['np'])*2*10
            dphase = float(self.ppdict['lpe']) / float(self.ppdict['nv'])*10
            dslice = float(self.ppdict['thk'])+float(self.ppdict['gap'])
            matrix = (int(self.ppdict['np'])/2,int(self.ppdict['nv']))
            dim = np.array([dread,dphase,dslice])
        if self.ppdict['apptype'] in ['im3D','im3Dshim']:
            dread = float(self.ppdict['lro']) / float(self.ppdict['np'])*2*10
            dphase = float(self.ppdict['lpe']) / float(self.ppdict['nv'])*10
            dphase2 = float(self.ppdict['lpe2']) / float(self.ppdict['nv2'])*10
            matrix = (int(self.ppdict['np'])/2,int(self.ppdict['nv']),int(self.ppdict['nv2']))
            dim = np.array([dread,dphase,dphase2])

        # making the Nifti affine and header

        self.aff = make_affine()
        self.hdr = make_header(self.aff)
#---------------------------------------------------------------------------------------------------
#                                            MAIN WRITE
#---------------------------------------------------------------------------------------------------


    def write(self,out):

        def save(out, hdr, aff):

            if '.nii' in out:
                out_name = str(out)
            else:
                out_name = str(out)+'.nii'
            img = nib.Nifti1Image(self.data, aff, hdr)
            #img.update_header()
            nib.save(img,out_name)
            os.system('gzip -f '+str(out_name))
            print('writeNifti : '+str(out_name)+' saved ... ')
                
        save(out,self.hdr, self.aff)
        

####################################################################################################
if __name__ == '__main__':

    print(    """
    Contains a class to write Nifti1 files from procpar and image data.
    Dimensions, orientations in the input data and procpar must match!
    
    INPUT:    procpar

        data = numpy.ndarray([phase, readout, slice, time])

    METHODS:
        write(out)

            nifti output dimensions are [phase, readout, slice, time]

            nifti affine is created from the procpar data

    """)

