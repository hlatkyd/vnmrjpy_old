#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

class niPlotter():
    """
    Take an ndarray and plot 3 perpendicular slices with
    the center point

    TODO: Take nifti as input and do the same?

    TODO: take aspect ratio prom procpar?
    """

    def __init__(self, data, procpar=None, axis=[0,1,2]):

        # check if data is 3D
        print('niPlotter: inpot data shape: '+str(data.shape))

        if type(data) is np.ndarray:

            if len(data.shape) == 3:
                self.data = data  # todo dynamic
            elif len(data.shape) == 4:
                self.data = data[...,0]

    def plot_slices(self):

        data = self.data
        img_1 = data[:,:,int(data.shape[2]/2)]
        img_2 = data[:,int(data.shape[1]/2),:]
        img_3 = data[int(data.shape[0]/2),:,:]
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(img_1)
        ax[1].imshow(img_2)
        ax[2].imshow(img_3)
        plt.show()

if __name__ == '__main__':

    data = np.random.rand(128,128,30)

    plotter = niPlotter(data)
    plotter.plot_slices()
