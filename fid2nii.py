#!/usr/bin/python3

import os
import sys
import numpy as np
import nibabel as nib
from readfid import fidReader
from readprocpar import procparReader
from writenifti import niftiWriter
from imake import imgSpaceMaker
from kmake import kSpaceMaker
from argparse import ArgumentParser
from shutil import copyfile

def fid2nii(indir,
            out=None,
            saveprocpar=True,
            save_kspace=False,
            save_imgspace=True):

    if not os.path.isdir(indir):
        raise(Exception('Please specify input fid directory'))

    if out == None:
        # if output is not given, take fid as basename for new dir
        out = indir[:-4]+'.nifti/'
        if not os.path.exists(out):
            os.makedirs(out)

    fid = indir + '/fid'
    procpar = indir + '/procpar'
    ppdict = procparReader(procpar).read()
    fid_data, fid_header = fidReader(fid, procpar).read()
    kspacemaker = kSpaceMaker(fid_data, procpar, fid_header)
    kspace = kspacemaker.make()
    imgspace = imgSpaceMaker(kspace, procpar).make() 

    kspace_real = [] # put separate channel data into separate lements of list
    kspace_imag = []

    for i in range(len(ppdict['rcvrs'])):
        kspace_real.append(np.real(kspace[i,...]))
        kspace_imag.append(np.imag(kspace[i,...]))
        writer1 = niftiWriter(procpar, kspace_real[-1])
        writer1.write(out+'kspace_real_ch'+str(i))
        writer2 = niftiWriter(procpar, kspace_imag[-1])
        writer2.write(out+'kspace_imag_ch'+str(i))

    sumimg = np.sum(np.absolute(imgspace), axis=0)
    writer_img = niftiWriter(procpar, sumimg)
    writer_img.write(out+'imgspace_sum')

    if saveprocpar == True:
        copyfile(procpar, out+'/procpar')

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('fid', help='input fid directory')
    parser.add_argument('out', help='output basename', nargs='?', default=None)
    args = parser.parse_args()

    fid2nii(args.fid, args.out)

