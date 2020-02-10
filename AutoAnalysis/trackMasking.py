import numpy as np
import matplotlib.pyplot as plt
import math
#from PixelDistribution import findPeakPosition


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



#Estimate Lamda
def calcLamda(distribution):
    integral = 0
    accuracy = 0.001

    firstPeak = max(distribution)

    i = 0
    while(integral < firstPeak or distribution[i]/integral > accuracy):
        integral = integral + distribution[i]
        i = i+1
    return -np.log(integral/sum(distribution))



#Estimate threshold
def calcThreshold(lamda,distribution,bins):

    sort = np.sort(distribution)
    index1 = np.argwhere(distribution == sort[-1])
    index2 = np.argwhere(distribution == sort[-2])

    x = 0
    cdf = 0
    while(cdf < (1-0.1/sum(distribution))):
        cdf = cdf + poisson(lamda,x)
        x = x+1
    return x*abs((bins[index2]-bins[index1]))


def poisson(lamda,x):
    if(x >= 0 and x%1 == 0):
        return math.exp(-lamda)*math.pow(lamda,x)/math.factorial(x)
    else:
        return 0