import numpy as np
import matplotlib.pyplot as plt
import trackMasking as tk
from astropy.io import fits
#import panda as pd



image = fits.getdata("C:/Users/95286/Documents/Python/DAMIC_code/FS_Avg_Img_27.fits")


normImage = image - np.median(image.reshape(image.size,1))

bins = np.arange(-40,1000)+0.5

xsize = image.shape[0]
ysize = image.shape[1]

print("normImage size=",normImage.size)
n,bins,patches = plt.hist(normImage.reshape(normImage.size,1),bins,density=False)
plt.close()




#mask
radius = 1

estimatedLamda = tk.calcLamda(n)
threshold = tk.calcThreshold(estimatedLamda,n,bins)
print('estimatedLamda =',estimatedLamda)
print('threshold = ',threshold)

mask1 = tk.mask(normImage,threshold,radius)
result = normImage[mask1]

maskedImage = normImage*1

for i in range(xsize):
    for j in range(ysize):
        if(mask1[i,j] == False):
            maskedImage[i,j] = max(normImage.flatten())


plt.subplot(2,1,1)
plt.imshow(normImage,cmap='Greys')
plt.colorbar()
plt.subplot(2,1,2)
plt.imshow(maskedImage,cmap='Greys')
plt.colorbar()



plt.figure()
plt.hist(normImage.flatten(),bins,density = False)
plt.xlabel("Pixel Value")
plt.ylabel("Counts")
plt.title("Raw Image")
plt.figure()
plt.hist(result.flatten(),bins,density = False)
plt.xlabel("Pixel Value")
plt.ylabel("Counts")
plt.title("Masked Image")
plt.show()