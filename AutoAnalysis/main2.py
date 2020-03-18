import numpy as np
import matplotlib.pyplot as plt
import trackMasking as tk
from astropy.io import fits
from scipy.stats import norm, poisson
import PixelDistribution as pd
import DamicImage
from PixelDistribution import findPeakPosition
import time



image = fits.getdata("C:/Users/95286/Documents/damic_images/FS_Avg_Img_30.fits")

temp = image*1

ts = np.arange(50)
Etime = np.zeros(50)

for j in range(1):
    start =time.perf_counter()
    image = np.vstack((image,temp))

    normImage = image[:-1,:] - np.median(image.reshape(image.size,1))

    bins = np.arange(-30,150)+0.5

    xsize = normImage.shape[0]
    ysize = normImage.shape[1]

    plt.figure()
    n,bins,patches = plt.hist(normImage.flatten(),bins=bins,density=False)



    maxi, mini = findPeakPosition(n,bins,nMovingAverage=4)
    separation = int(maxi[1]-maxi[0])
    peak1 = int(maxi[0]+0.5)


    #mask
    radius = 2

    fitLamda = tk.calcLamda(n, bins ,normImage.size)
    result = normImage*1








    # plt.figure()
    threshold = tk.calcThreshold(fitLamda,result.size,n,bins)
    mask1 = tk.mask(normImage,threshold,radius)
    result = normImage[mask1]
    n,bins,patches = plt.hist(result.flatten(),bins=bins,density=False)
    fitMu, fitSigma, N1 = tk.normalFit(n,bins,separation)
    fitLamda, volume, c2dof = tk.lsFit(n,bins,separation,threshold)
    plt.close()

    end = time.perf_counter()

    Etime[j] = end - start
    # print(end-start)





print("separation = ", separation)
print("radius = ", radius)
print("threshold = ",threshold)
print("fitLamda = ",fitLamda)
print("fitSigma = ", fitSigma)
print("\u03C7^2 per degree of freedom = ", c2dof)
print("Number of masked pixels = ",normImage.size - result.size)
print("Total Number = ", normImage.size)

volume = volume/np.sqrt(2*np.pi*fitSigma[0]**2)






x = np.linspace(-30,30,1000)
y = np.arange(peak1,100,separation)
plt.plot(x, N1[0]*norm.pdf(x,fitMu[0],fitSigma[0]), 'orange')
plt.plot(x, N1[1]*norm.pdf(x,fitMu[1],fitSigma[1]), 'orange')
plt.plot(x, N1[2]*norm.pdf(x,fitMu[2],fitSigma[2]), 'orange')
plt.plot(y,volume*poisson.pmf((y-peak1)/separation,fitLamda),'red', lw=0.9)
plt.xlabel('Pixel Value')
plt.ylabel('Counts')

# plt.yscale('log')
# plt.ylim(0.1, 10000)


plt.figure()
plt.plot(ts,Etime)
plt.xlabel("n times original image")
plt.ylabel("time")
plt.show()





