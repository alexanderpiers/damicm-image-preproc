import numpy as np
import scipy.stats
import scipy.optimize as optimize
import readFits
import matplotlib.pyplot as plt

def imageEntropy(image):
	"""
	Computes the shannon information entropy (-plog(p)) of the image. Uses the pixel distribution to create the probability distribution
	Inputs:
		image - (nrows x ncolumns) 2D-numpy array (though function accepts any shape numpy array)
	Outputs:
		entropy - double of the entropy given the image pixel distribution
	"""

	# Create a histogram of the pixel values
	pixelBins = np.arange(np.min(image), np.max(image) + 1)
	pixelVals, _ = np.histogram(image.flatten(), bins=pixelBins)

	# Compute the entropy
	pixelProbabilityDistribution = pixelVals / np.sum(pixelVals)
	entropy = np.sum([-p * np.log(p) for p in pixelProbabilityDistribution if p > 0])

	return entropy

def imageEntropySlope(skImage, maxNskips=30):
	"""
	Computes the rate of change of entropy as a function of the number of skips
	Inputs:
		skImage - (nrow x ncolumns x nskips) 3D numpy array
	Output:
		dSdSkips - dS/dSkips is the slope of the entropy (fit data to linear function)
	"""

	if len(skImage.shape) < 3 or skImage.shape[-1] == 1:
		return (-1, -1, -1)

	nskips = skImage.shape[2]
	nskips = np.min([nskips, maxNskips])
	singleImageEntropy = np.zeros(nskips)

	for i in range(nskips):
		singleImageEntropy[i] = imageEntropy(skImage[:,:,i])

	# Fit with linear regression
	linfunc = lambda x, *p: p[0] + p[1] * x

	paramGuess = [np.mean(singleImageEntropy), 0]
	paramFit, paramCov = optimize.curve_fit(linfunc, np.arange(0, nskips), singleImageEntropy, p0=paramGuess )

	# Return slope and uncertainy
	return paramFit[1]*1e3, np.sqrt(np.diag(paramCov))[1]*1e3, singleImageEntropy

def computeImageNoise(skImage, maxNskips=50):
	"""
		Computes the noise of the image by fitting the zero electron peak to a gaussian
	"""

	# Figure out how many skips in the image
	if len(skImage.shape) == 2:
		nskips = 1
		skImage = np.expand_dims(skImage, 2)
	else:
		nskips = skImage.shape[-1]

	nskips = np.min([nskips, maxNskips])

	# Define gaussian fit
	gausfunc = lambda x, *p: p[0] * np.exp( -( x - p[1] )**2 / ( 2 * p[2]**2 ))

	# Iterate over each image in the skipped image, fit Guassian to the noise
	imageNoiseVec = []
	for i in range(nskips):

		# Create histogram of pixels
		med = np.median(skImage[:, :, i])
		mad = scipy.stats.median_absolute_deviation(skImage[:, :, i], axis=None)
		bins = np.arange(med - 3*mad, med+3*mad)
		y, xedges = np.histogram(skImage[:,:,i], bins=bins)
		xcenter = xedges[:-1] + np.diff(xedges)[0]

		# Perform fit
		paramGuess = [np.sum(y) / np.sqrt(2 * np.pi * mad**2), med, mad]
		paramOpt, _ = optimize.curve_fit(gausfunc, xcenter, y, p0=paramGuess)

		# Append to vector
		imageNoiseVec.append(paramOpt[2])

	return np.mean(imageNoiseVec)

def convertValErrToString(param):
	"""
		Converts a param, err tuple to string with +/- between terms
		Inputs:
			param - (val, err) combination of a given fit parameter
		Output:
			paramString - "val +/- err"
	"""
	return "%.2g +/- %.2g" % (param[0], param[1])

if __name__ == '__main__':
	
	filename = "../Img_20.fits"

	_, data = readFits.read(filename)

	# Test entopy slope
	# slope, err, entropy = imageEntropySlope(data[:,:,:25])
	# print(convertValErrToString((slope, err)))

	# plt.plot(entropy, "*k")

	# Test noise fit
	print("Test Image noise")
	print(computeImageNoise(data[:,:,:-1], maxNskips=25))

	# plt.show()


