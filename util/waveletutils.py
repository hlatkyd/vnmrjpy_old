#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import pywt


WAVELET = 'db2'
MODE = 'per'
LEVEL = 5

#UPDATE:

# THIS IS UTTER SHIT
# pywt.coeffs_to_array(coeffs) and imshow does this normally

# keeping for the wall of shame

# TODO complex display is not too okayish...

# UPDATE OF UPDATE : maybe not too bad?


def imshow_wt2d(coeffs, cmap='gray', fam='db', mode='per', \
                vmin=0, vmax=255):
    """
    Arrange wavelet transform coefficients and plot in a sensible manner
    Uses matplotlib
    """
    def _stack(current, next_coeffs, normalize=True):
        """
        Extend image by stacking the coeffs
        next_coeffs = (horizontal, vertical, diagonal)
        should be used recursively in case of multiple levels
        """
        if normalize:
            nex_coeffs = [i/(i.max()/255) for i in next_coeffs]
        h1 = np.hstack([current, next_coeffs[0]])
        h2 = np.hstack([next_coeffs[1], next_coeffs[2]])
        return np.vstack([h1,h2])

    #print('Wavelet coefficients: {}'.format(len(coeffs)))
    len_x = [coeffs[0].shape[0]]
    len_y = [coeffs[0].shape[1]]
    for i in range(1,len(coeffs)):
        len_x.append(coeffs[i][0].shape[0])
        len_y.append(coeffs[i][1].shape[1])
    len_x_sum = sum(len_x)
    len_y_sum = sum(len_y)
    #print('Wavelet coefficients: {}'.format(len(coeffs)))
    #print('X shape: {} ; y shape: {}'.format(len_x, len_y))
    #print('X sum: {} ; y sum: {}'.format(len_x_sum, len_y_sum))

    #image = np.zeros([len_x_sum, len_y_sum],dtype=coeffs[0].dtype)
    # arranging decomp coeffs

    img = coeffs[0]
    img = img / (img.max()/255)
    for i in range(1,len(coeffs)):

        img = _stack(img,coeffs[i])

    plt.imshow(np.absolute(img), cmap=cmap, vmin=vmin, vmax=vmax)

if __name__ == '__main__':

    image = pywt.data.camera()
    print('image shape: {}'.format(image.shape))
    coeffs = pywt.wavedec2(image, WAVELET, level=LEVEL, mode=MODE)
    recon_img = pywt.waverec2(coeffs, WAVELET, MODE)
    print(type(coeffs))
    print(type(coeffs[3]))
    print(len(coeffs[1]))
    print(coeffs[3][0].shape)
    print(type(coeffs[0]))
    print(coeffs[0].shape)

    imshow_wt2d(coeffs)
    plt.show()
