import numpy as np
import matplotlib.pyplot as plt
import math




#Create mask
def mask(image,threshold,xradius,yradius):
    xsize = image.shape[0]
    ysize = image.shape[1]
    mask1 = np.ones((xsize, ysize),dtype = bool)
    for ii in range(xsize):
        for jj in range(ysize):
            if(image[ii,jj] >= threshold):
                if(ii < xradius):
                    if(jj < yradius):
                        mask1[:(ii+xradius+1),:(jj+yradius+1)] = False
                    elif((ysize - jj) <= yradius):
                        mask1[:(ii+xradius+1),(jj-yradius):] = False
                    else:
                        mask1[:(ii+xradius+1),(jj-yradius):(jj+yradius+1)] = False
                elif((xsize - ii) <= xradius):
                    if(jj < yradius):
                        mask1[(ii-xradius):,:(jj+yradius+1)] = False
                    elif((ysize - jj) <= yradius):
                        mask1[(ii-xradius):,(jj-yradius):] = False
                    else:
                        mask1[(ii-xradius):,(jj-yradius):(jj+yradius+1)] = False
                else:
                    if(jj < yradius):
                        mask1[(ii-xradius):(ii+xradius+1),:(jj+yradius+1)] = False
                    elif((ysize - jj) <= yradius):
                        mask1[(ii-xradius):(ii+xradius+1),(jj-yradius):] = False
                    else:
                        mask1[(ii-xradius):(ii+xradius+1),(jj-yradius):(jj+yradius+1)] = False
    return mask1

