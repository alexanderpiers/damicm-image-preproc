import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import palettable
import argparse
import scipy.stats

sys.path.append("/home/damic/Documents/damicm-image-preproc/AutoAnalysis")

import PoissonGausFit as pgf 
import readFits
import DamicImage
import lmfit

if __name__ == '__main__':
	


	parser = argparse.ArgumentParser(description="Plot Skipper Image")
	parser.add_argument("-f", "--filename", help="Skipper FITS file to plot")
	parser.add_argument("-n", "--noise", action="store_true", help="Plot noise as a function of number of skips")
	parser.add_argument("-r", "--reverse", action="store_true", help="Parity flip of the histogram")
	parser.add_argument("-k", "--conversion", default=-7, help="ADU / e- Conversion value")
	parser.add_argument("-l", "--lta", action="store_true", help="LTA data format")
	args = parser.parse_args()

	# distribute command line args to variables
	filename = args.filename
	plotNoise = args.noise
	reverse = args.reverse
	aduConversion = float(args.conversion)
	isLTA = args.lta
	
	# Read data

	if(isLTA):
		ext = 4
		header, data = readFits.readLTA(filename)
		header = header[ext]
		nskips = int(header["NSAMP"])

		data = readFits.reshapeLTAData(data[ext], int(header["NROW"]), int(header["NCOL"]), nskips)
	else:
		header, data = readFits.read(filename)

	nskips = int(header["NSAMP"])

	# 1nt(data)
	bw = 100
	skoffset = 10
	# print(np.mean(data[40:,5:,skoffset:-1], axis=-1))
	dataPositionCuts = data[5:, 5:, skoffset:-1]
	print(dataPositionCuts.shape)
	img = DamicImage.DamicImage(np.mean(dataPositionCuts, axis=-1), bw=bw, reverse=reverse)

	print(img.image.shape)
	minres = pgf.computeGausPoissDist(img, aduConversion=aduConversion, npoisson=30, darkCurrent=-0.1, sigma=-scipy.stats.median_absolute_deviation(dataPositionCuts, axis=None) / np.sqrt(nskips) / 2)
	params = minres.params
	print(lmfit.fit_report(minres))

	medSubImage = (1, -1)[reverse] * (img.image - params["offset"])

	# Use fit information to make an educated guess on how much to mask.
	nElectronMask = 30
	maskThreshold = nElectronMask * params["ADU"]
	maskImage = DamicImage.MaskedImage(medSubImage, bw=bw, minRange=1000, reverse=False, maskThreshold=maskThreshold, maskRadiusX=1, maskRadiusY=0)
	print(maskImage.med)
	print(maskImage.mad)
	

	minres = pgf.computeGausPoissDist(maskImage, aduConversion=aduConversion, npoisson=60, darkCurrent=-0.4, sigma=-scipy.stats.median_absolute_deviation(dataPositionCuts, axis=None) / np.sqrt(nskips))
	params = minres.params


	print(lmfit.fit_report(minres))


	fig, ax = plt.subplots(3, 1, figsize=(12, 10))
	offset = img.med - 5 * img.mad
	colorGradient = palettable.cmocean.sequential.Amp_20.mpl_colormap
	cax = ax[0].imshow(img.image, aspect="auto",  cmap=colorGradient, vmin=img.med-3*img.mad, vmax=img.med+3*img.mad)
	fig.colorbar(cax, ax=ax[0])
	ax[0].set_xlabel("x [pixels]", fontsize=14)
	ax[0].set_ylabel("y [pixels]", fontsize=14)	



	# ax[1].hist(img.centers, bins=img.edges, weights=img.hpix) # Plot histogram of data
	ax[1].errorbar(maskImage.centers, maskImage.hpix, yerr=np.sqrt(maskImage.hpix), fmt="ok", markersize=3, alpha=0.7)

	# Plot fit results
	par = pgf.paramsToList(params)
	x = np.linspace(maskImage.centers[0], maskImage.centers[-1], 2000)
	ax[1].plot(x, pgf.fGausPoisson(x, *par), "--r", linewidth=3)
	ax[1].set_xlabel("Pixel Value", fontsize=14)
	ax[1].set_yscale("log")
	ax[1].set_ylim(0.05, params["N"] / 2)
	# ax[1].set_xlim(maskImage.centers[maskImage.hpix > 0][0] - 10, maskImage.edges[-1])
	fig.suptitle(filename, fontsize=14)
	ax[1].legend([r"$\sigma$=%.2f e-, $\lambda$=%.2f e- / pix / exposure"%(params["sigma"].value / params["ADU"].value, params["lamb"].value)], fontsize=16)

	
	ax[2].imshow(maskImage.mask.astype(int), aspect="auto", cmap="gray", vmin=0, vmax=1)
	ax[2].set_xlabel("x [pixels]", fontsize=14)
	ax[2].set_ylabel("y [pixels]", fontsize=14)	

	# cax0 = ax[0].imshow(data[20:,:,1], aspect="auto", cmap=colorGradient, vmin=img.med-3*img.mad, vmax=img.med+3*img.mad)
	# cax1 = ax[1].imshow(data[20:,:,100] - data[20:,:,4], aspect="auto", cmap=colorGradient, vmin=-3*img.mad, vmax=3*img.mad)
	# fig.colorbar(cax0, ax=ax[0])
	# fig.colorbar(cax1, ax=ax[1])

	# fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
	# # ax1.hist(data[:,:,:-1].flatten(), bins=img.edges)
	# # ax1.set_yscale("log")
	# # ax1.plot(data[50, 100, :], "k", linewidth=2)
	# # ax1.plot(data[20, 100, :], "r", linewidth=2)
	# ax1.hist(data[:,:,5].flatten(), bins=img.edges, color="r", alpha=0.3)
	# # ax1.hist(np.mean(data[:,:,5:], axis=-1).flatten(), bins=img.edges, color="b", alpha=0.3)
	# ax1.set_yscale("log")

	
	if plotNoise:
		# nskips = header["NDCMS"]
		print(nskips)
		npoints = 20
		skipsToAverage = np.round(np.logspace(np.log10(skoffset), np.log10(nskips), npoints)).astype("int")

		skipperResolution = []
		skipperResolutionErr = []
		aduConversion = params["ADU"].value
		skipsToAverage = np.unique(skipsToAverage)
		for i, sk in enumerate(skipsToAverage):

			img = DamicImage.DamicImage(np.mean(data[:,:,skoffset:sk+1], axis=-1), reverse=reverse, )
			print(data[:,:,skoffset:sk+1].shape)
			lmmin = pgf.computeGausPoissDist(img, aduConversion=aduConversion, npoisson=50, darkCurrent=params["lamb"].value)
			skipperResolution.append(np.abs(pgf.paramsToList(lmmin.params)[0]))
			skipperResolutionErr.append(pgf.parseFitMinimum(lmmin)["sigma"][1])
			# print(lmfit.fit_report(lmmin))

		print(skipperResolution)
		print(skipperResolutionErr)
		figR, axR = plt.subplots(1, 1)
		axR.plot(skipsToAverage-skoffset+1, skipperResolution, "o", color="k")
		axR.plot(skipsToAverage-skoffset+1, skipperResolution[0] / np.sqrt(skipsToAverage-skoffset+1), "--r", linewidth=2)
		print(skipperResolution)
		print(skipsToAverage)
		# fitparam = scipy.optimize.curve_fit(reso, skipsToAverage, skipperResolution)
		# print(fitparam)
		axR.set_yscale("log")
		axR.set_xscale("log")
		axR.set_xlabel("Number of Skips", fontsize=14)
		axR.set_ylabel("Resolution [ADU]", fontsize=14)

	plt.show()
