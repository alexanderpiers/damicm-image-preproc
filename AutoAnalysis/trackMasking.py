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
                if(ii < yradius):
                    if(jj < xradius):
                        mask1[:(ii+yradius+1),:(jj+xradius+1)] = False
                    elif((ysize - jj) <= xradius):
                        mask1[:(ii+yradius+1),(jj-xradius):] = False
                    else:
                        mask1[:(ii+yradius+1),(jj-xradius):(jj+xradius+1)] = False
                elif((xsize - ii) <= yradius):
                    if(jj < xradius):
                        mask1[(ii-yradius):,:(jj+xradius+1)] = False
                    elif((ysize - jj) <= xradius):
                        mask1[(ii-yradius):,(jj-xradius):] = False
                    else:
                        mask1[(ii-yradius):,(jj-xradius):(jj+xradius+1)] = False
                else:
                    if(jj < xradius):
                        mask1[(ii-yradius):(ii+yradius+1),:(jj+xradius+1)] = False
                    elif((ysize - jj) <= yradius):
                        mask1[(ii-yradius):(ii+yradius+1),(jj-xradius):] = False
                    else:
                        mask1[(ii-yradius):(ii+yradius+1),(jj-xradius):(jj+xradius+1)] = False
    return mask1

