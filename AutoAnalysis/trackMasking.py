import numpy as np
import matplotlib.pyplot as plt
import math
from PixelDistribution import findPeakPosition
import scipy.stats as sta


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
def calcLamda(distribution, totalNum):
    integral = 0
    accuracy = 0.001

    maxi,mini = findPeakPosition(distribution,bins)
    index = np.argwhere(bins == maxi[0])

    i = 0
    while(i < index or distribution[i]/integral > accuracy):
        integral = integral + distribution[i]
        i = i+1
    return -np.log(integral/totalNum)



#Estimate threshold
def calcThreshold(lamda,totalNum,distribution,bins):

    maxi,mini = findPeakPosition(distribution,bins)
    index1 = np.argwhere(bins = maxi[0])
    index2 = np.argwhere(bins = mini[1])

    x = 0
    while(sta.cdf(x,lamda) < (1-0.1/totalNum)):
        x = x+1
    print("x,",x)

    return x*abs((bins[index2]-bins[index1])) + bins[index1]


# def poisson(lamda,x):
#     if(x >= 0 and x%1 == 0):
#         return math.exp(-lamda)*math.pow(lamda,x)/math.factorial(x)
#     else:
#         return 0