#!/usr/bin/python3

import sys
import os
sys.path.append('/home/david/dev/vnmrjpy')
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from kmake import kSpaceMaker
from imake import imgSpaceMaker
from argparse import ArgumentParser
from readfid import fidReader
from readprocpar import procparReader
from writenifti import niftiWriter
from writefdf import fdfWriter


"""
Intention of script:
---------------------------------------------------------------------
Wrapper of vnmrjpy in place of Xrecon, if custom recon is used.
Depending on sequence 'execproc' a MAGICAL 
macro calls this script via shell. Example: GEMS: im2D('proc') -> 
im2Dxrecon('recon') : line 56-57: shell('pyrecon {params}')

What it does:
-------------------------------------------------------------------
Takes fid in CUREXP/acqfil/fid as input
Gives .fdf files in CUREXP/recon/* as output
        Maybe: nifti as well

Vnmrj should be able to handle the resulting fdf files for display, etc..

How it does this:
-------------------------------------------------------------------
fid is read in with fidReader, kspace is made with kSpaceMaker.
K-space data and img spacedat are stored in numpy.ndarray([])
image is made with imgSpaceMaker, and some part of the 
reconstructed data (example: combined magnitude) is
written to fdf files with fdfWriter
"""

CUREXP = '/home/david/vnmrsys/exp2'

# TODO CUREXP should be dynamic to work on other accounts/operators
# ex.: search from 'global' on current account or something,,,

class Pyrecon():

    def __init__(self, curexp):
        self.curexp = curexp
        self.fid = curexp + '/acqfil/fid'
        self.procpar = curexp + '/procpar'
        #self.outdir = curexp + '/recon' # final
        self.outdir = curexp + '/recon' # testing

    def run(self):

        def combine_magnitude(image_data):

            fdf_data = np.sum(np.absolute(image_data), axis=0)

            return fdf_data

        fidreader = fidReader(self.fid, self.procpar)        
        fid_data, fid_header = fidreader.read()
       
        kmaker = kSpaceMaker(fid_data, self.procpar, fid_header)
        kspace = kmaker.make() 

        #TODO here comes the choice for cs params: skipint, skiptab
        # image = numpy.ndarray([coil, phase, readout, slice, echo-or-rep])
        # image is dtype=complex

        imaker = imgSpaceMaker(kspace, self.procpar, skiptab=None, skipint=None)
        image = imaker.make()

        #TODO what should be written to fdf? (no coil param, or should be?)
        # fdfdata = numpy.ndarray([phase, read, slice, echo-or-rep])

        fdf_data = combine_magnitude(image)

        # make recon dir if not exists
        #if not os.path.isdir(self.outdir):
        #    os.
   

        # fid path is only for header naming purposes 
        fdfwriter = fdfWriter(fdf_data, self.procpar, self.fid)
        fdfwriter.write(self.outdir)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('input_curexp', nargs='?', default=None)
    args = parser.parse_args()
    if args.input_curexp != None:
        recon = Pyrecon(args.input_curexp)
    else:
        recon = Pyrecon(CUREXP)
    recon.run()













