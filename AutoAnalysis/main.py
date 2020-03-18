import numpy as np
import matplotlib.pyplot as plt
import trackMasking as tk
from astropy.io import fits
from scipy.stats import norm, poisson
from PixelDistribution import findPeakPosition
from scipy.ndimage import label




image = fits.getdata("C:/Users/95286/Documents/Python/DAMIC_code/FS_Avg_Img_27.fits")


normImage = image[1:25,:] - np.median(image.reshape(image.size,1))

bins = np.arange(-30,150)+0.5

xsize = normImage.shape[0]
ysize = normImage.shape[1]

n,bins,patches = plt.hist(normImage.flatten(),bins,density=False)
plt.close()

maxi, mini = findPeakPosition(n,bins,nMovingAverage=4)
separation = int(maxi[1]-maxi[0])
#mask
radius = 1

fitLamda = tk.calcLamda(n, bins ,normImage.size)
result = normImage*1

for i in range(1):
    threshold = tk.calcThreshold(fitLamda,result.size,n,bins)
    mask1 = tk.mask(normImage,threshold,radius)
    result = normImage[mask1]
    n,bins,patches = plt.hist(result.flatten(),bins,density=False)
    fitMu, fitSigma, N1 = tk.normalFit(n,bins)
    fitLamda, N2, c2dof = tk.lsFit(n,bins,separation,threshold)

print(' threshold = ',threshold)



maskedImage = normImage*1
maxval = max(normImage.flatten())
pattern = np.zeros((xsize,ysize))
for i in range(xsize):
    for j in range(ysize):
        if(mask1[i,j] == False):
            maskedImage[i,j] = maxval
            pattern[i,j] = 1


x = label(pattern)
print("Number of Clusters",x[1])



# Image plot
plt.subplot(2,1,1)
plt.imshow(normImage,cmap='Greys',aspect='auto')
plt.colorbar()
plt.subplot(2,1,2)
plt.imshow(maskedImage,cmap='Greys',aspect='auto')
plt.colorbar()


# Raw Image
plt.figure()
plt.hist(normImage.flatten(),bins,density = False)
plt.xlabel("Pixel Value")
plt.ylabel("Counts")
plt.title("Raw Image")
#plt.yscale('log')


# Masked Image and Fits

N2 = N2/np.sqrt(2*np.pi)/fitSigma[0]


peak1 = int(maxi[0]+0.5)

print("fit Mu = ",fitMu)
print('fit Sigma = ', fitSigma)
print('fitLamda = ', fitLamda)

x = np.linspace(-30,30,1000)
y = np.arange(peak1,100,separation)
plt.figure()
p1 = plt.hist(result.flatten(),bins,density = False)
plt.plot(x, N1[0]*norm.pdf(x,fitMu[0],fitSigma[0]), 'orange')
plt.plot(x, N1[1]*norm.pdf(x,fitMu[1],fitSigma[1]), 'orange')
plt.plot(x, N1[2]*norm.pdf(x,fitMu[2],fitSigma[2]), 'orange')
plt.plot(y,N2*poisson.pmf((y-peak1)/separation,fitLamda),'red', lw=0.9)


plt.xlabel("Pixel Value")
plt.ylabel("Counts")
plt.title("Masked Image")
# plt.yscale('log')
# plt.ylim(0.1,7000)

plt.text(75,5000,"fit $\lambda$: "+str(round(fitLamda,3)))
plt.text(75,4500,"threshold: "+str(round(int(threshold),3)))

plt.show()