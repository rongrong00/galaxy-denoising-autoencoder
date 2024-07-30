################################################
# add_background.py
#
# Python script to combine the simulated image data with the sky background into injected images
#################################################

import numpy as np
from astropy.io import fits
import os
#from astropy.convolution import convolve

dataReadPath = './simulated_galaxy_images/' #path to the simulated galaxy files from galfit
dataWritePath = './injected_galaxy_images/' #path to store the output injected image files
backgroundPath = './cutout_clean/'  #path to the sky background images

NUM_FILES_TOTAL = 1448
IMG_WIDTH = IMG_HEIGHT = 512 #No. of pixels length/width -wise in each cutout

data_list = sorted(os.listdir(backgroundPath))
files = [i for i in data_list if i.endswith('.fits')]
for i in range(len(data_list)):
    img_data = fits.getdata(dataReadPath+"output_img_"+str(i)+".fits",memmap=False)
    background_data = fits.getdata(backgroundPath + files[i])
    combined_data = img_data + background_data #Combine the simulated image data with the background 
    hdu = fits.PrimaryHDU(combined_data)
    hdu.writeto(dataWritePath+"output_img_"+str(i)+".fits",overwrite=True) 