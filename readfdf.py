#!/usr/bin/python3

import os
import glob
import numpy as np
import nibabel as nib

from argparse import ArgumentParser
import matplotlib.pyplot as plt
from readprocpar import procparReader

class fdfReader():
    """
    Class fdfReader(input,output,procpar=None)
    METHODS:

    read() : reads fdf binaries in .img directory, or single fdf files

        return ({fdf header dict}, data=np.ndarray)

                data = np.ndarray([receivers, time, slice, readout, phase])        # legacy        
                

                data = np.ndarray([receivers, phase, readout, slice, time])    


    """

    def __init__(self, inPath, procpar=None, printlines=False):
        self.path = inPath
        self.procpar = procpar
        self.printlines = printlines
    
    def read(self):

        #---------------- auxiliary functions for read method -----------------

        def preproc_fdf(fdf):

            with open(fdf,'rb') as openFdf:
                fdata = bytearray(openFdf.read())
                nul = fdata.find(b'\x00')
                header = fdata[:nul]
                data = fdata[nul+1:]
            return (header,data)

        # ----------parse fdf header and return into a dictionary --------------

        def parse_header(header):
            keys_to_parse = sorted(['rank','roi','location','spatial_rank',\
                            'matrix','orientation',\
                            'studyid','gap','pe_size','ro_size',\
                            'pe2_size', 'abscissa',\
                            'storage'])
            to_delete = ('char','float','int')
            header = header.decode('ascii').split('\n')
            header_dict = {}    
            for line in header:             # some formatting of header
                if self.printlines:
                    print(line)
                for item in to_delete:
                    if line.startswith(item):
                        line = line.split(item,1)
                        break
                try:
                    line = line[1].lstrip()
                    line = line.lstrip('*').rstrip(';')
                    if '[]' in line:
                        line = line.replace('[]','')
                    if '{' in line:
                        line = line.replace('{','(')
                    if '}' in line:
                        line = line.replace('}',')')
                    if ' ' in line:
                        line = line.replace(' ','')
                    line = line.split('=')
                    header_dict[line[0]] = line[1]        
                except:
                    continue

            for item in keys_to_parse:
                if item in header_dict.keys():            
                    if item == 'abscissa':
                        tempval = header_dict[item][1:-1];''.join(tempval)
                        tempval = tempval.replace('"','')                
                        header_dict[item] = tuple([k for k in tempval.split(',')])
                    if item == 'matrix':            
                        tempval = header_dict[item][1:-1];''.join(tempval)                
                        header_dict[item] = tuple([int(k) for k in tempval.split(',')])
                    if item == 'roi':
                        tempval = header_dict[item][1:-1];''.join(tempval)                
                        header_dict[item] = tuple([float(k) for k in tempval.split(',')])
                    if item == 'ro_size' or item == 'pe_size' or item == 'pe2_size':
                        header_dict[item] = int(header_dict[item])
                    if item == 'storage':
                        tempval = header_dict[item];''.join(tempval)
                        tempval = tempval.replace('"','')
                        header_dict[item] = str(tempval)
                    if item == 'orientation':
                        tempval = header_dict[item][1:-1];''.join(tempval)                
                        header_dict[item] = tuple([float(k) for k in tempval.split(',')])
                    if item == 'location':
                        tempval = header_dict[item][1:-1];''.join(tempval)                
                        header_dict[item] = tuple([float(k) for k in tempval.split(',')])
                    if item == 'gap':
                        header_dict[item] = float(header_dict[item])
                    if item == 'slices':
                        header_dict[item] = int(header_dict[item])
                    if item == 'TR':
                        header_dict[item] = float(header_dict[item])/1000
            return header_dict

        #----------process bynary data based on header--------------------
    
        def prepare_data(binary_data):

            matrix = self.header_dict['matrix']

            if self.header_dict['storage'] == 'float' and \
               self.header_dict['bits'] == '32':
                dt = np.dtype('float32'); dt = dt.newbyteorder('<')

            else:
                print('')
                print('error: data type incorrectly specified in "prepare_data"\n')
                return -1

            img_data = np.frombuffer(binary_data, dtype=dt)
            img_data = np.reshape(img_data,matrix)
            return img_data
        #--------------------------------------------------------------------------------------
        #                                     main read method
        #--------------------------------------------------------------------------------------
        if os.path.isdir(self.path):
            self.procpar = str(self.path)+'/procpar'
            fdf_list = sorted(glob.glob(str(self.path)+'/*.fdf'))
            (header, data) = preproc_fdf(fdf_list[0]) # run preproc once to get header
            ppr = procparReader(self.procpar)
            self.ppdict = ppr.read()
        else:
            try:            
                ppr = procparReader(self.procpar)
                self.ppdict = ppr.read()
            except:
                print('\nfdfReader.read() warning : Please specify procpar file!\n')
            fdf_list =[self.path]
            (header, data) = preproc_fdf(self.path)
            
        self.header_dict = parse_header(header)
        # ------------------------process if 3d -------------------------
        if self.header_dict['spatial_rank'] == '"3dfov"':

            full_data = []
            time_concat = []
            time = len([1 for i in fdf_list if 'slab001' in i])
            for i in fdf_list: # only 1 item, but there migh be more in future
                                
                (header, data) = preproc_fdf(i)
                img_data = prepare_data(data)
                full_data.append(img_data) # full data in one list

            self.data_array = np.asarray(full_data)
            #self.data_array = np.swapaxes(self.data_array, 0,3)

        #------- -----------------process if 2d------------------------------------
        elif self.header_dict['spatial_rank'] == '"2dfov"':

            full_data = []
            time_concat = []
            time = len([1 for i in fdf_list if 'slice001' in i])

            for i in fdf_list:                
                (header, data) = preproc_fdf(i)
                img_data = prepare_data(data)
                img_data = np.expand_dims(np.expand_dims(img_data,2),3) # expand 2d to 4d
                full_data.append(img_data) # full data in one list
            # make sublists
            slice_list =[full_data[i:i+time] for i in range(0,len(full_data),time)]
            for aslice in slice_list:
                time_concat.append(np.concatenate(tuple(aslice),axis=3))
            slice_time_concat = np.concatenate(time_concat, axis=2) #slice+time concatenated

 
            self.data_array = slice_time_concat

            if str(self.ppdict['orient']) == 'trans90':  # vomit inducing method to properly flip data

                print('found orientation')
                self.data_array = np.swapaxes(self.data_array, 0, 1)

    


        # --------------------process if 1d------------------------ TODO ????
        elif self.header_dict['spatial_rank'] == '"1dfov"':
            pass
        

        return (self.header_dict, self.data_array)
        #--------------------------------------------------------------------------------------
        #                            END of main read method
        #--------------------------------------------------------------------------------------

    def print_header(self):

        for key in self.header_dict.keys():
            print(key+' : '+str(self.header_dict[key]))
        

#----------utility for testing ---------------------------
if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('input',help='input .fdf file or .img directory')
    parser.add_argument('output',help='output file')
    args = parser.parse_args()

    rdr = fdfReader(args.input, printlines=True)
    (hdr, data) = rdr.read()
    #rdr.print_header()
    #print(data.shape)
    
