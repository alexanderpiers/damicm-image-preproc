import numpy as np
import matplotlib.pyplot as plt
import math
import CCDMean


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
def calcLamda(distribution,peak,totalNum):
    n0 = 0
    accuracy = 0.001
    i = 0
    while(n0 < peak or distribution[i]/n0 > accuracy):
        n0 = n0 + distribution[i]
        i = i+1
    return -np.log(n0/totalNum)

#Could refine
def approxPeak(image,position):
    peak = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(image[i,j] > position-0.1 and image[i,j] < position+0.1):
                peak = peak + 1
    return peak



#Estimate threshold
def calcThreshold(lamda,totalNum):
    x = 0
    sum = 0
    while(sum < (1-0.1/totalNum)):
        sum = sum + poisson(lamda,x)
        x = x+1
    return (x-1)*10


def poisson(lamda,x):
    if(x >= 0 and x%1 == 0):
        return math.exp(-lamda)*math.pow(lamda,x)/math.factorial(x)
    else:
        return 0