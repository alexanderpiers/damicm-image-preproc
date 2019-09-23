import sys
import numpy as np
from astropy.io import fits
import palettable

def read(filename):
	"""
	Reads the fits file from filename and returns the header information and 
	a numpy array of the data (nrows, ncolumns, nskips)
	"""

	fitsImg = fits.open(filename)

	# Get numpy of rows, columns, and skips
	header = fitsImg[0].header
	nrows = header["NAXIS2"]
	nskips = header["NDCMS"]
	ncolumns = header["NAXIS1"] // nskips

	# Get data and put it in the shape we want
	data = fitsImg[0].data
	data = np.reshape(data, (nrows, nskips, ncolumns), "F")
	data = np.transpose(data, (0, 2, 1))

	fitsImg.close()
	return header, data



if __name__ == '__main__':
	# Testing the read function
	fitsFilename = "../../2019-09-16/Img_8.fits"
	header, data = read(fitsFilename)
	colors = palettable.scientific.sequential.Devon_20.mpl_colormap

	for key, val in header.items():
		print(key + ": " + str(val))

	import matplotlib.pyplot as plt

	img = np.mean(data[:,:800,5:], -1)
	imgMean = np.mean(img)
	imgStd = np.std(img)
	fig, ax = plt.subplots(1, 1, figsize=(12, 8))
	cax = ax.imshow(img, aspect="auto", vmin=imgMean - 3*imgStd, vmax=imgMean + 3*imgStd, cmap=colors)

	# n = 10
	# fig, axs = plt.subplots(n, n)
	# print(axs.shape)
	# for i in range(axs.size):
	# 	xi = i // n
	# 	yi = i % n
	# 	img = data[:,:,i]
	# 	imgMean = np.mean(img)
	# 	imgStd = np.std(img)
	# 	cax = axs[xi,yi].imshow(img, aspect="auto", vmin=imgMean - 3*imgStd, vmax=imgMean + 3*imgStd, cmap=colors)
	fig.colorbar(cax, ax=ax)

	# Image histogram
	figH, axH = plt.subplots(1, 1, figsize=(12, 8))
	bins = np.linspace(imgMean - 3*imgStd, imgMean + 3*imgStd, 100)
	axH.hist(img.flatten(), bins=bins)
	axH.set_xlabel("Pixel Value", fontsize=16)
	axH.set_yscale("log")

	print(imgMean)
	print(imgStd)
	plt.show()

