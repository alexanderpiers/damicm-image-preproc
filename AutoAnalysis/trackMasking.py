import numpy as np
import matplotlib.pyplot as plt
import math




#Create mask
def mask(image,threshold,radius):
    xsize = image.shape[0]
    ysize = image.shape[1]
    mask1 = np.ones((xsize, ysize),dtype = bool)
    for ii in range(xsize):
        for jj in range(ysize):
            if(image[ii,jj] >= threshold):
                if(ii < radius):
                    if(jj < radius):
                        mask1[:(ii+radius+1),:(jj+radius+1)] = False
                    elif((ysize - jj) <= radius):
                        mask1[:(ii+radius+1),(jj-radius):] = False
                    else:
                        mask1[:(ii+radius+1),(jj-radius):(jj+radius+1)] = False
                elif((xsize - ii) <= radius):
                    if(jj < radius):
                        mask1[(ii-radius):,:(jj+radius+1)] = False
                    elif((ysize - jj) <= radius):
                        mask1[(ii-radius):,(jj-radius):] = False
                    else:
                        mask1[(ii-radius):,(jj-radius):(jj+radius+1)] = False
                else:
                    if(jj < radius):
                        mask1[(ii-radius):(ii+radius+1),:(jj+radius+1)] = False
                    elif((ysize - jj) <= radius):
                        mask1[(ii-radius):(ii+radius+1),(jj-radius):] = False
                    else:
                        mask1[(ii-radius):(ii+radius+1),(jj-radius):(jj+radius+1)] = False
    return mask1

