import numpy as np
import matplotlib.pyplot as plt
import math
from PixelDistribution import findPeakPosition
import scipy.stats as sta
from scipy.optimize import curve_fit



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
def calcLamda(distribution,bins,totalNum):

    maxi,mini = findPeakPosition(distribution,bins,nMovingAverage=4)
    indexMini = np.argwhere(bins == mini[0])
    integral = peakVolume(0,indexMini,distribution)

    return -np.log(integral/totalNum)







#Estimate threshold
def calcThreshold(lamda,totalNum,distribution,bins):

    accuracy = 0.1

    maxi,mini = findPeakPosition(distribution,bins, nMovingAverage=4)
    x = 0
    while(sta.poisson.cdf(x,lamda) < (1-accuracy/totalNum)):
        x = x+1

    return x*abs(maxi[1]-maxi[0]) + maxi[0]


def normalFit(distribution,bins,separation):
    maxi,mini = findPeakPosition(distribution,bins,nMovingAverage=4)
    mini1 = np.argwhere(bins == maxi[0])[0,0] + int(separation/2)
    central = bins+0.5

    # print("maxi = ",maxi)
    # print("mini=",mini)
    index = [0,mini1,mini1+separation,mini1+2*separation]


    mu = np.zeros(3)
    sig = np.zeros(3)
    volume = np.zeros(3)


    for i in range(3):
        a = central[index[i]:index[i+1]]
        volume[i] = peakVolume(index[i],index[i+1],distribution)
        para, pcov = curve_fit(gaussian,a,distribution[index[i]:index[i+1]]/volume[i],[maxi[0]+i*separation,1])
        mu[i] = para[0]
        sig[i] = para[1]


    return mu, sig, volume


def lsFit(distribution,bins,separation,threshold):

    maxi, mini = findPeakPosition(distribution, bins, nMovingAverage=4)
    indexMini = np.argwhere(bins == maxi[0])[0,0] + int(separation/2)

    num = int(threshold/separation)+1
    index = [0,indexMini]
    peaks = np.zeros(num)

    for i in range(1,num):
        if(i > 1):
            index.append(i*separation+indexMini)
        peaks[i-1] = peakVolume(index[i-1],index[i],distribution)


    volume = np.sum(peaks)
    a = np.arange(num)
    lamb,pcov = curve_fit(poisson,a,peaks/volume)


    chi = np.zeros(num)
    trueVals = np.zeros(num)
    for i in range(num):
        trueVals[i] = round(int(volume*sta.poisson.pmf(i,lamb)))
        chi[i] = (peaks[i]-volume*sta.poisson.pmf(i,lamb))**2/(volume*sta.poisson.pmf(i,lamb))
    # print("chi = ",chi)
    # print("peaks = ", peaks)
    # print("trueVal = ", trueVals)
    c2dof = sum(chi)/(num-2)

    return float(lamb),volume,c2dof


def peakVolume(lower,upper,distribution):
    integral = 0
    i = lower
    while(i < upper):
        integral = integral + distribution[i]
        i = i+1

    return integral





def poisson(x,lamda):
    return sta.poisson.pmf(x,lamda)

def gaussian(x,mu,sig):
    return np.exp(-0.5*(x-mu)**2/sig**2)/np.sqrt(2*np.pi*sig**2)








