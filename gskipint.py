#!/usr/bin/python3

import numpy as np
import nibabel as nib
from scipy.signal import gaussian
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from writenifti import niftiWriter
from readprocpar import procparReader
#TODO give procpar as input and generate mask from that

class skipintGenerator():
    """
    Compressed sensing sequences will use 'skipint' vnmrj parameter as in
    original Agilent sequence 'ge3d_elliptical'. This program 
    generates text files containing this parameter according to
    desired k-space size, and other stuff...

    result: K space center kept, etc etc....
    
    Should be able to generate in 2D, 2Dmultislice,  3D, 4D.
    Should be uncorrelated in slice and time
    """
    #np.random.seed(400)

    def __init__(self,\
                dim=None,\
                shape=None,\
                reduction=4,\
                slices=1,\
                time=1,\
                center=1/16,
                procpar=None):

        self.procpar = procpar
        self.reduction = float(reduction)
        self.center = center

        if procpar == None:

            self.dim = int(dim)
            self.shape = shape  # (pe, ro) or (pe, pe2, ro) 
            self.slices = slices
            self.time = time  #TODO get this from procpar as well

        if dim==None or shape==None:
        
            ppdict = procparReader(procpar).read()
            
            self.time = time #TODO make this dynamic

            try:
                self.shape = (int(ppdict['nv']),
                                int(ppdict['np'])//2,
                                int(ppdict['ns']))
                self.dim = 2
                self.slices = int(ppdict['ns'])
                print('Making 2D slices of shape: {}'.format(self.shape))
            except:
                try:
                
                    self.shape = (int(ppdict['nv']),
                                    int(ppdict['np'])//2,
                                    int(ppdict['nv2']))
                    self.dim = 3
                    print('Making 3D volume of shape: {}'.format(self.shape))
                except:
                    raise(Exception("procpar dictionary error while making pars"))

    def generate_kspace_mask(self):
        """
        This is for CS testing and generates K-space type output
        """
        def make_2d_mask(num):

            mask2d = np.zeros((self.shape[0], self.shape[1]))

            weights = list(gaussian(self.shape[0], self.shape[0]/6))

            points = np.zeros(self.shape[0])
            center_ones = [int(points.shape[0]/2*(1-self.center)),\
                            int(points.shape[0]/2*(1+self.center))]
            center_ones_num = center_ones[1] - center_ones[0]
            points[center_ones[0]:center_ones[1]] = 1
            num = num - center_ones_num
            if num < 0:
                raise(Exception('No remaining points to choose: too high\
                                 reduction, or too much points in center'))
        
            
            indices_available = [i for i, x in enumerate(points) if x == 0]
            weights_available = [weights[i] for i in indices_available]

            weights_available_normalized = [float(i)/sum(weights_available) \
                                            for i in weights_available]

            choices = np.random.choice(indices_available,\
                                        int(num),\
                                        p=weights_available_normalized,\
                                        replace=False)

            for index in choices:
                points[index] = 1

            for i in range(self.shape[1]):

                mask2d[:,i] = np.array(points)

            return mask2d

        def make_3d_mask(num):

            def gaussian_2d(x, y, stddev_x, stddev_y):

                gx = gaussian(x, stddev_x)
                gy = gaussian(y, stddev_y)

                weights = np.outer(gx,gy.T)
                return weights

            def make_circle_mask(x, y, rx, ry, cx=None, cy=None):

                center_ones = np.zeros([x,y])
                grid = np.ogrid[-x/2 : x/2, -y/2 : y/2]
                mask = grid[0]**2 + grid[1]**2 <= rx*ry
                center_ones[mask] = 1
                return center_ones


            mask3d = np.zeros(self.shape)
            points_2d = np.zeros([self.shape[0], self.shape[1]])

            weights_2d = gaussian_2d(self.shape[0], self.shape[1],\
                            self.shape[0]/6, self.shape[1]/6)

            radius_x = self.shape[0]*self.center
            radius_y = self.shape[1]*self.center
            center_ones = make_circle_mask(self.shape[0],
                                            self.shape[1],
                                            radius_x,
                                            radius_y)

            # reshaping weights, points into 1d array
            weights_1d = np.reshape(weights_2d, weights_2d.size)
            points_1d = np.reshape(center_ones, points_2d.size)

            num = num - np.count_nonzero(center_ones == 1)
            if num < 0:
                raise(Exception('No remaining points to choose: too high\
                                 reduction, or too much points in center'))
        
            
            indices_available = [i for i, x in enumerate(points_1d) if x == 0]
            weights_available = [weights_1d[i] for i in indices_available]

            weights_available_normalized = [float(i)/sum(weights_available) \
                                            for i in weights_available]

            choices = np.random.choice(indices_available,\
                                        int(num),\
                                        p=weights_available_normalized,\
                                        replace=False)

            for index in choices:
                points_1d[index] = 1


            points_2d = np.reshape(points_1d, points_2d.shape)

            for i in range(self.shape[2]):

                mask3d[:,:,i] = np.array(points_2d)

            return mask3d

        # --------------------- generate k mask MAIN ------------------------


        if self.dim == 2:

            num = np.floor(self.shape[0]/self.reduction)
            self.kspace_mask = np.zeros([self.shape[0], self.shape[1], 
                                        self.slices, self.time])
            for t in range(self.time):

                for slc in range(self.slices):

                    self.kspace_mask[:,:,slc,t] = make_2d_mask(num)

            print('Mask ready, shape: {}'.format(self.kspace_mask.shape))

        elif self.dim ==3:

            self.kspace_mask = np.zeros([self.shape[0], self.shape[1], 
                                        self.shape[2], self.time])
            num = np.floor(self.shape[0]*self.shape[1]/self.reduction)

            for t in range(self.time):

                self.kspace_mask[:,:,:,t] = make_3d_mask(num)

        else:
    
            pass

    def genereate_from_procpar(self):

        pass

    def save_to_text(self, out):

        pass

    def save_to_nifti(self, out):

        if self.procpar == None:

            if self.dim == 2:

                aff = np.eye(len(self.kspace_mask.shape))
                aff = aff*np.array([1,1,10,1])
                img = nib.Nifti1Image(self.kspace_mask,aff)
                nib.save(img, out) 

            if self.dim == 3:

                aff = np.eye(len(self.kspace_mask.shape))
                aff = aff*np.array([1,1,1,1])
                img = nib.Nifti1Image(self.kspace_mask,aff)
                nib.save(img, out) 

        else:
            
            writer = niftiWriter(self.procpar, self.kspace_mask)
            writer.write(out)

    def plot_check(self):

        plt.imshow(self.kspace_mask[:,:,0,0])
        plt.show()

if __name__ == '__main__':
    #--------------------------------------------------------------------------
    parser = ArgumentParser()
    parser.add_argument('--shape',
                        help='kspace shape',
                        nargs='*',
                        metavar=' x, y, z',
                        type=int)
    parser.add_argument('--reduction','-r', help='reduction factor, ex : 2, 3, 4')
    parser.add_argument('--plot','-p',\
                        help='option to plot results', action='store_true')
    parser.add_argument('--slices','-s',\
                        help='number of slices', nargs='?', default=1, type=int)
                        
    parser.add_argument('--time','-t',\
                        help='number of TRs', nargs='?', default=1, type=int)
    parser.add_argument('--procpar',\
                        help='give procpar if you want to generate from procpar data',\
                         nargs='?', default=None)
    parser.add_argument('--output',\
                        help='output path',\
                         nargs='?', default='gskipint_output')

    args = parser.parse_args()
    print(args)

    #--------------------------------------------------------------------------
    if args.procpar != None:
        # generate from procpar data    
        gen = skipintGenerator(procpar=args.procpar, reduction=float(args.reduction))

    else:
        # if procpar is not given generate manually
        if len(args.shape) == 3:
            dim = 3
            shape = (args.shape[0], args.shape[1], args.shape[2])
        elif len(args.shape) == 2:

            dim = 2
            shape = (args.shape[0], args.shape[1])

        gen = skipintGenerator(dim,\
                                shape,\
                                slices=int(args.slices),\
                                time=int(args.time),\
                                reduction=float(args.reduction))
    
    gen.generate_kspace_mask()
    gen.save_to_nifti(args.output)
    if args.plot:
        gen.plot_check()


