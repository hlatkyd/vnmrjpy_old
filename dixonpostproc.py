#!/usr/bin/python3

import numpy as np
import nibabel as nib
import sys
import os
import glob
import csv
from argparse import ArgumentParser
from shutil import copyfile
"""

"""
DIR_DIXON_PROC = '/proc_dixon'  # dixon process directory
DIR_QUANT_PROC = '/proc_quant'  # T2, T2* etc process dir
DIR_OUT = '/proc_textoutput'
WHOLEMASK = 'fieldmap_mask.nii.gz'
LIVERMASK = 'liver_mask.nii.gz'
VISCERALMASK = 'hasmask.nii.gz'

def check_validity(study):

    if os.path.isdir(study):
        if os.path.isdir(study+DIR_DIXON_PROC):
            print('dixonpostproc : Directory check ... /proc_dixon OK')
        else:
            raise(Exception('dixonpostproc : Directory check ... \
                            No dixon procdir found'))
        if os.path.isdir(study+DIR_QUANT_PROC):
            print('dixonpostproc : Directory check ... /proc_quant OK')
        else:
            raise(Exception('dixonpostproc : Directory check ... \
                            No quantitative mapping procdir found'))
    else:
        raise(Exception('Study directory not found'))

def apply_fat_water_shift_correction():
    """
    fix shifted pixels as of recent paper...
    """
    pass

def apply_whole_body_mask():
    # TODO
    # do this before the T2 fittings, for god's sake
    for i in glob.glob(STUDY+DIR_QUANT_PROC+'/m*t2map.nii.gz'):
        if 'masked' not in i:
            command = 'fslmaths '+i+' -mul '+STUDY+DIR_QUANT_PROC+'/'+WHOLEMASK+' '\
                        +i[:-7]+'_masked'
            os.system(command)

def apply_liver_mask():

    for i in glob.glob(STUDY+DIR_QUANT_PROC+'/m*t2map.nii.gz'):
        if 'masked' and 'liver' not in i:
            command = 'fslmaths '+i+' -mul '+STUDY+DIR_QUANT_PROC+'/'+LIVERMASK+' '\
                        +i[:-7]+'_liver'
            os.system(command)

    for i in [STUDY+DIR_DIXON_PROC+'/fat.nii.gz', \
                STUDY+DIR_DIXON_PROC+'/water.nii.gz']:
        command = 'fslmaths '+i+' -mul '+STUDY+DIR_DIXON_PROC+'/'+LIVERMASK+' '\
                    +i[:-7]+'_liver'

def apply_visceral_mask():

    for i in [STUDY+DIR_DIXON_PROC+'/fat.nii.gz', \
                STUDY+DIR_DIXON_PROC+'/water.nii.gz']:
        command = 'fslmaths '+i+' -mul '+STUDY+DIR_DIXON_PROC+'/'+VISCERALMASK+' '\
                    +i[:-7]+'_visceral'
        os.system(command)

def get_sum(nifti):

    data = nib.load(nifti).get_fdata()
    sum_ = np.sum(data[data != 0])
    return sum_

def get_mean(nifti):

    data = nib.load(nifti).get_fdata()
    mean = np.mean(data[data != 0])
    return mean

def mask_liver(nifti):

    pass


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('study', help='Study directory')
    args = parser.parse_args()

    STUDY = args.study

    STUDY_BASE = STUDY.rsplit('/')[-1]
    check_validity(STUDY)

    # making sure masks exist in proc_quant:

    try:
        copyfile(STUDY+DIR_DIXON_PROC+'/'+WHOLEMASK, \
               STUDY+DIR_QUANT_PROC + '/'+WHOLEMASK)
        copyfile(STUDY+DIR_DIXON_PROC+'/'+LIVERMASK, \
                STUDY+DIR_QUANT_PROC + '/'+LIVERMASK)
        copyfile(STUDY+DIR_DIXON_PROC+'/'+VISCERALMASK, \
                STUDY+DIR_QUANT_PROC + '/'+VISCERALMASK)
    except:
        print('Warning, mask not found')

    # making text output dir

    if not os.path.isdir(STUDY+DIR_OUT):
        os.mkdir(STUDY+DIR_OUT)
    else:
        print('Text output dir exists already')

    # multiply stuff with the masks:

    apply_whole_body_mask()

    apply_liver_mask()

    apply_visceral_mask()

    # TODO apply_fat_water_shift_correction()

    # CALCULATING SUMS AND AVERAGES

    water_sum = get_sum(STUDY+DIR_DIXON_PROC+'/water.nii.gz')
    fat_sum = get_sum(STUDY+DIR_DIXON_PROC+'/fat.nii.gz')

    visceral_fat_sum = get_sum(STUDY+DIR_DIXON_PROC+'/fat_visceral.nii.gz')    
    visceral_water_sum = get_sum(STUDY+DIR_DIXON_PROC+'/water_visceral.nii.gz')    

    water_liver = get_sum(STUDY+DIR_DIXON_PROC+'/water_liver.nii.gz')
    fat_liver = get_sum(STUDY+DIR_DIXON_PROC+'/fat_liver.nii.gz')

    # TODO filenaming conventions should be set sometime....

    t2_liver = get_mean(STUDY+DIR_QUANT_PROC+'/mems_'+STUDY_BASE+'_01_t2map_liver.nii.gz') 
    t2star_liver = get_mean(STUDY+DIR_QUANT_PROC+'/mgems_'+STUDY_BASE+'_01_t2map_liver.nii.gz')
 
    fat_ratio_whole = fat_sum / (water_sum + fat_sum)
    fat_ratio_liver = fat_liver / (water_liver + fat_liver)
    fat_visceral_per_whole = (visceral_fat_sum - fat_liver) / fat_sum

    # writing numbers to textfile

    txtfile = STUDY+DIR_OUT+'/'+STUDY_BASE+'_dixon_proc.txt'
    lines = []
    lines.append('whole_water,{0:.2f}\n'.format(water_sum))
    lines.append('whole_fat,{0:.2f}\n'.format(fat_sum))
    lines.append('whole_fat_ratio,{0:.2f}\n'.format(fat_ratio_whole))
    lines.append('fat_visceral_per_whole,{0:.2f}\n'.format(fat_visceral_per_whole))
    lines.append('liver_water,{0:.2f}\n'.format(water_liver))
    lines.append('liver_fat,{0:.2f}\n'.format(fat_liver))
    lines.append('liver_fat_ratio,{0:.2f}\n'.format(fat_ratio_liver))
    lines.append('liver_t2_mean,{0:.4f}\n'.format(t2_liver))
    lines.append('liver_t2star_mean,{0:.4f}\n'.format(t2star_liver))
    with open(txtfile,'w') as openfile:
        openfile.write(''.join(lines))
        




