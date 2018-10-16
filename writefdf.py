#!/usr/local/bin/python3.6

import sys
import os
import struct
sys.path.append('/home/david/dev/vnmrjpy')
import numpy as np
from readprocpar import procparReader
from argparse import ArgumentParser
import nibabel as nib

class fdfWriter():
    """
    Class to write data to .fdf files readable by Vnmrj for display
    Intended use: run on custum procesing pipeline and take part in writing
    fdf files from fid data in 'curexp' dir in -study time-
    Intended as an accessory only for this use    

    fdf header is in C syntax.
    header is disgustingly hard coded .... 
    """
    def __init__(self, data, procpar, fid_path=None):
        """
        data = numpy.ndarray(phase, read, slice, echo)
    
        procpar = /path/to/procpar
        """
        self.fid = fid_path
        self.procpar = procpar
        self.p = procparReader(procpar).read()
        self.data = data

    def write(self, outdir):

        """
        FDF header definitions are in Vnmrj User Programming
        Reference manual
        """
        try:
            echoes = int(self.p['ne'])
        except:
            echoes = 1

        images = int(self.data.shape[3])//echoes
        slices = int(self.data.shape[2])
        echoes = int(self.p['ne'])

        def write_fdf(outfile, cropped_data):
            """
            Writes single fdf file.

            outfile = path/to/file

            cropped_data = np.ndarray(phase, read, slice, echo)
            """


            # --------------Setting header --------------------------
            
            # fdf should start with magic number string
            magic_number = '#!/usr/local/fdf/startup'
            
            # setting other header parameters
            if self.p['seqfil'] in ['gems','sems','fsems','mgems','epip']:

                _spatial_rank = '2dfov'
                _rank = 2
                _matrix = (int(self.p['np'])//2, int(self.p['nv']))
                _slices = int(self.p['ns'])
                _ro_size = int(self.p['np'])//2
                _pe_size = int(self.p['nv'])
                _gap = float(self.p['gap'])
                _span = (float(self.p['lro']), float(self.p['lpe']))
                _roi = (float(self.p['lro']),\
                        float(self.p['lpe']),\
                        float(self.p['gap'])+float(self.p['thk']))
    
                _abscissa = '{"cm", "cm"}'

                _ordinate = 'intensity'
                _origin = '{0.0,0.0}'
            elif self.p['seqfil'] in ['ge3d', 'ge3d_elliptical']:

                _spatial_rank = '3dfov'
                _rank = 3
                _matrix = (int(self.p['np'])//2,\
                            int(self.p['nv']),\
                            int(self.p['nv2']))
                _ro_size = int(self.p['np'])//2
                _pe_size = int(self.p['nv'])
                _pe2_size = int(self.p['nv2'])

                _span = (float(self.p['lro']),\
                        float(self.p['lpe']),\
                        float(self.p['lpe2']))

                _roi = (float(self.p['lro']),\
                        float(self.p['lpe']),\
                        float(self.p['lpe2']))
                _imagescale = float(1)
                #TODO
                _gap = float(self.p['gap'])
                _slab_no = 1
                _slabs = 1
                _abscissa = '{"cm", "cm", "cm"}'
                _origin = '{0.0,0.0,0.0}'
                _ordinate = '{ "intensity" }'
            else:
                raise Exception('fdfwriter: sequence not implemented yet')

            _TE = float(self.p['te'])*1000
            _te = float(self.p['te'])
            _TR = float(self.p['tr'])*1000
            _tr = float(self.p['tr'])
            _storage = 'float'
            _bits = 32
            _type = 'absval'
            _bigendian = 0
            _sequence = str(self.p['seqfil'])
            #_studyid = str(self.p['studyid'])
            _studyid = 's_2016052401'  # TODO there is some bug....
            _fid = str(self.fid)

            # dunno if needed...

            _display_order = 0
            _image = 1.0

            # TODO
            _orientation = '{0.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,0.0}'
            _location = '{0.0,0.0,0.0}'
            _nucleus = '{"H1","C13"}'
            _nucfreq = '{399.459406,100.453412}'

            # ------------------- Making header --------------------

            if self.p['seqfil'] in ['gems','sems','fsems','mgems','epip']:

                header_lines = []
                header_lines.append(magic_number)
                header_lines.append('float\trank = {};'.format(_rank))
                header_lines.append('char\t*storage = "{}";'.format(_storage))
                header_lines.append('char\t*type = "{}";'.format(_type))
                header_lines.append('float\tbits = {};'.format(_bits))
                header_lines.append('float\tmatrix[] = {};'.format(_matrix).\
                                    replace('(','{').replace(')','}'))
                header_lines.append('float\tspan[] = {};'.format(_span).\
                                    replace('(','{').replace(')','}'))
                header_lines.append('float\troi[] = {};'.format(_roi).\
                                    replace('(','{').replace(')','}'))
                header_lines.append('float\tro_size = {};'.format(_ro_size))
                header_lines.append('float\tpe_size = {};'.format(_pe_size))
                header_lines.append('float\tgap = {};'.format(_gap))
                header_lines.append('int\tbigendian = {};'.format(_bigendian))
                header_lines.append('char\t*spatial_rank = "{}";'.format(_spatial_rank))
                header_lines.append('char\t*sequence = "{}";'.format(_sequence))
                header_lines.append('char\t*studyid = "{}";'.format(_studyid))
                header_lines.append('char\t*file = "{}";'.format(_fid))
                # maybe TODO

                header_lines.append('int\tdisplay_order = "{}";'.format(_display_order))
                header_lines.append('float\timage = "{}";'.format(_image))

                #TODO
                header_lines.append('float\torientation[] = {};'.format(_orientation))
                header_lines.append('float\tlocation[] = {};'.format(_location))
                header_lines.append('char\t*nucleus[] = {};'.format(_nucleus))
                header_lines.append('char\t*abscissa[] = {};'.format(_abscissa))
                header_lines.append('float\tnucfreq[] = {};'.format(_nucfreq))
                header_lines.append('float\torigin[] = {};'.format(_origin))
                header_lines.append('float\tnucfreq[] = {};'.format(_nucfreq))

            elif self.p['seqfil'] in ['ge3d','ge3d_elliptical', 'mprage3d']:

                header_lines = []
                header_lines.append(magic_number)
                header_lines.append('float\trank = {};'.format(_rank))
                header_lines.append('char\t*storage = "{}";'.format(_storage))
                header_lines.append('char\t*type = "{}";'.format(_type))
                header_lines.append('float\tbits = {};'.format(_bits))
                header_lines.append('float\tmatrix[] = {};'.format(_matrix).\
                                    replace('(','{').replace(')','}'))
                header_lines.append('float\tspan[] = {};'.format(_span).\
                                    replace('(','{').replace(')','}'))
                header_lines.append('float\troi[] = {};'.format(_roi).\
                                    replace('(','{').replace(')','}'))
                header_lines.append('float\tro_size = {};'.format(_ro_size))
                header_lines.append('float\tpe_size = {};'.format(_pe_size))
                header_lines.append('float\tpe2_size = {};'.format(_pe2_size))
                header_lines.append('int\tbigendian = {};'.format(_bigendian))
                header_lines.append('char\t*spatial_rank = "{}";'.format(_spatial_rank))
                header_lines.append('char\t*sequence = "{}";'.format(_sequence))
                header_lines.append('char\t*studyid = "{}";'.format(_studyid))
                header_lines.append('char\t*file = "{}";'.format(_fid))
                header_lines.append('float\tgap = {};'.format(_gap))
                header_lines.append('int\tslabs = {};'.format(_slabs))
                header_lines.append('int\tslab_no = {};'.format(_slab_no))
                # maybe TODO

                header_lines.append('float\timagescale = {};'.format(_imagescale))

                #TODO
                header_lines.append('float\torientation[] = {};'.format(_orientation))
                header_lines.append('float\tlocation[] = {};'.format(_location))
                header_lines.append('float\troi[] = {};'.format(_roi))
                header_lines.append('char\t*nucleus[] = {};'.format(_nucleus))
                header_lines.append('char\t*abscissa[] = {};'.format(_abscissa))
                header_lines.append('float\tnucfreq[] = {};'.format(_nucfreq))
                header_lines.append('float\torigin[] = {};'.format(_origin))
                header_lines.append('char\tordinate[] = {};'.format(_ordinate))

            else:
                raise(Exception('fdfWriter: sequence not implemented yet'))

            # ----------------- Making binary data --------------------
            float_data = np.float32(cropped_data).tobytes()
            print(type(float_data))
            #binary_data = struct.pack('<f',b'\x00'+float_data)
            binary_data =b'\x00'+float_data

            # -------------------Writing single fdf----------------------

            with open(outfile,'w') as openfile:
                header = '\n'.join(header_lines)
                openfile.write(header)           

            #binary_data = np.float32(cropped_data).tobytes()

            with open(outfile,'ab') as openfile:

                #openfile.write(b'\x00')
                openfile.write(binary_data)
 

        # ------------------------- WRITE MAIN --------------------------------
        try: 
            os.mkdir(outdir)
        except:
            print('Warning directory {} exists'.format(outdir))
        os.system('cp '+str(self.procpar)+' '+str(outdir)+'/procpar')

        zn = '000'  # helper string
        
        if self.p['seqfil'] in ['gems','sems','fsems','mgems','epip']:

            for i_slc in range(slices):

                for j_imag in range(images):

                    for k_echo in range(echoes):

                        str_slc = zn[:-len(str(i_slc+1))]+str(i_slc+1)
                        str_imag = zn[:-len(str(j_imag+1))]+str(j_imag+1)
                        str_echo = zn[:-len(str(k_echo+1))]+str(k_echo+1)
                        file_name = 'slice'+str_slc+\
                                    'image'+str_imag+\
                                    'echo'+str_echo+'.fdf'
                        file_path = outdir+'/'+file_name
                        try:
                            open(file_path,'x')
                        except:
                            print('Warning: file {} already exists.\n \
                                    Rewriting...'.format(file_path))
                        write_fdf(file_path,\
                                self.data[:,:,i_slc,j_imag*(k_echo+1)+k_echo])


        elif self.p['seqfil'] in ['ge3d','ge3d_elliptical']:

            print('images: {}, echoes: {}'.format(images, echoes))

            for j_imag in range(images):

                for k_echo in range(echoes):

                    str_imag = zn[:-len(str(j_imag+1))]+str(j_imag+1)
                    str_echo = zn[:-len(str(k_echo+1))]+str(k_echo+1)
                    file_name = 'slab001'+\
                                'image'+str_imag+\
                                'echo'+str_echo+'.fdf'
                    file_path = outdir+'/'+file_name
                    try:
                        open(file_path,'x')
                    except:
                        print('Warning: file {} already exists.\n \
                                Rewriting...'.format(file_path))
                    write_fdf(file_path,\
                                self.data[:,:,:,j_imag*(k_echo+1)+k_echo])
                    print('self data shape: {}'.format(self.data.shape))
                    print('Saving 3D fdf: Done')
        else:
            raise(Exception('writeFdf: sequence not implemented'))

if __name__ == '__main__':

    #parser = ArgumentParser()
    #parser.add_argument('inputfile')

    TESTNIFTI = 'tempdata/imake_test.nii.gz'
    TESTPP = 'tempdata/kmake_test.procpar'

    TESTOUT_DIR = 'tempdata/writefdfdir.img'

    FID_PATH = 'tempdata/sems_20160527_01.fid'

    data = nib.load(TESTNIFTI).get_fdata()

    fdfw = fdfWriter(data, TESTPP, FID_PATH)
    fdfw.write(TESTOUT_DIR)

