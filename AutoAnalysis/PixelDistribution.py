import numpy as np
import scipy.stats
import scipy.optimize as optimize
import readFits
import matplotlib.pyplot as plt
import PixelStats as ps


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
        singleImageEntropy[i] = imageEntropy(skImage[:, :, i])

    # Fit with linear regression
    linfunc = lambda x, *p: p[0] + p[1] * x

    paramGuess = [np.mean(singleImageEntropy), 0]
    paramFit, paramCov = optimize.curve_fit(
        linfunc, np.arange(0, nskips), singleImageEntropy, p0=paramGuess
    )

    # Return slope and uncertainy
    return paramFit[1] * 1e3, np.sqrt(np.diag(paramCov))[1] * 1e3, singleImageEntropy


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
    gausfunc = lambda x, *p: p[0] * np.exp(-(x - p[1]) ** 2 / (2 * p[2] ** 2))

    # Iterate over each image in the skipped image, fit Guassian to the noise
    imageNoiseVec = []
    for i in range(nskips):

        # Create histogram of pixels
        med, mad = estimateDistributionParameters(skImage[:, :, i])
        bins = np.arange(med - 3 * mad, med + 3 * mad)
        y, xedges = np.histogram(skImage[:, :, i], bins=bins)
        xcenter = xedges[:-1] + np.diff(xedges)[0]

        # Perform fit
        paramGuess = [np.sum(y) / np.sqrt(2 * np.pi * mad ** 2), med, mad]
        paramOpt, _ = optimize.curve_fit(gausfunc, xcenter, y, p0=paramGuess)

        # Append to vector
        imageNoiseVec.append(paramOpt[2])

    return np.mean(imageNoiseVec)


def computeSkImageNoise(image, nMovingAverage=10):
    """
	Computes the noise on single electron measurements. Takes a skipper image, searches for single electron peaks
	and fits a gaussian function to it
	Inputs:
		image - (nrows, ncolumns) 2D numpy array that we are trying to find the single electron noise of
		nMovingAverage - int, number of points to use in the smoothing moving average
	Outputs:
		skImageNoise - double, standard deviation of the gaussian fit to a single peak
		skImageNoiseErr - double, fit error of the standard deviations
	"""

    # Histogram the image
    hSkipper, bincSkipper, _ = histogramImage(image, nsigma=6, minRange=100)

    # Smooth Data
    smoothHSkipper = np.convolve(
        hSkipper, np.ones(nMovingAverage) / nMovingAverage, mode="same"
    )

    # Peak find on smoothed data
    derivative = np.diff(smoothHSkipper)
    maximaIndex = np.nonzero(
        np.hstack((0, (derivative[:-1] > 0) * (derivative[1:] < 0), 0))
    )
    minimaIndex = np.nonzero(
        np.hstack((0, (derivative[:-1] < 0) * (derivative[1:] > 0), 0))
    )
    maximaLoc = bincSkipper[maximaIndex]
    minimaLoc = bincSkipper[minimaIndex]
    
    # fig, ax = plt.subplots(1, 1, figsize=(12,9))
    # ax.plot(bincSkipper, hSkipper, "*k")
    # ax.plot(bincSkipper, smoothHSkipper, "--r")
    # ax.plot(maximaLoc, hSkipper[maximaIndex], "*b")
    # ax.plot(minimaLoc, hSkipper[minimaIndex], "*g")

    # Find the fit range (choose the appropriate maxima and minima)
    try:
        fitMin = minimaLoc[-2]
        fitMax = minimaLoc[-1]
        fitMean = maximaLoc[-2] if maximaLoc[-1] > minimaLoc[-1] else maximaLoc[-1]
    except:
	    print("could not find enough minima and maxima")
	    return -1, -1

    # Keep only data in the fit range
    fitIndex = (bincSkipper >= fitMin) * (bincSkipper <= fitMax)
    fitXRange = bincSkipper[np.nonzero(fitIndex)]
    fitSkipperValues = hSkipper[np.nonzero(fitIndex)]

    # Fit peak with gaussian
    gausfunc = lambda x, *p: p[0] * np.exp(-(x - p[1]) ** 2 / (2 * p[2] ** 2))
    paramGuess = [
        hSkipper[np.nonzero(bincSkipper == fitMean)][0],
        fitMean,
        (fitMax - fitMin) / 6,
    ]
    try:
        paramOpt, cov = optimize.curve_fit(
            gausfunc, fitXRange, fitSkipperValues, p0=paramGuess
        )
    except (RuntimeError, optimize.OptimizeWarning) as e:
        return -1, -1

    # x = np.linspace(fitMin, fitMax, 100)
    # ax.plot(x, gausfunc(x, *paramOpt), 'k', linewidth=3)

    # Return noise and error
    skImageNoise = paramOpt[2]
    skImageNoiseErr = np.sqrt(cov[2, 2])
    return skImageNoise, skImageNoiseErr


def computeImageTailRatio(image, nsigma=4.):
    """
	Calculates the ratio of the number of pixels in the left tail of the distribution to the number expected if it was
	just gaussian noise
	Inputs:
		image - (nrows, ncols, [nskips]) numpy array. Should be raw images and not the combined image
		nsigma - double, threshold definition of the tail
	Outputs:
		tailRatio - double, ratio of actual to expected number of points in the tail of the variance distribution. >> 1 is a proxy for tracks

	"""

    imageH, imageBinCenters, imageBinEdges = histogramImage(image, nsigma=nsigma)

    # Fit cluster variance to a gaussian
    gausfunc = lambda x, *p: p[0] * np.exp(-(x - p[1]) ** 2 / (2 * p[2] ** 2))
    sigmaGuess = (imageBinEdges[-1] - imageBinEdges[0]) / (2 * nsigma)
    paramGuess = [
        np.sum(imageH) / np.sqrt(2 * np.pi * sigmaGuess ** 2),
        imageBinCenters[imageBinCenters.size // 2],
        sigmaGuess,
    ]
    paramOpt, paramCov = optimize.curve_fit(
        gausfunc, imageBinCenters, imageH, p0=paramGuess
    )

    # Compute the ratio between the tails of the fit and data
    tailLoc = paramOpt[1] - nsigma * paramOpt[2]
    tailRatio = np.sum(image < tailLoc) / (
        image.size
        * (scipy.stats.norm.cdf(tailLoc, loc=paramOpt[1], scale=paramOpt[2]))
    )

    return tailRatio


def convertValErrToString(param):
    """
		Converts a param, err tuple to string with +/- between terms
		Inputs:
			param - (val, err) combination of a given fit parameter
		Output:
			paramString - "val +/- err"
	"""
    return "%.2g +/- %.2g" % (param[0], param[1])


def histogramImage(image, nsigma=3, minRange=None):
    """
	Creates a histogram of an image (or any data set) of a reasonable range with integer (ADU) spaced bins
	Inputs:
		image - ndarray containing data values to be histogrammed
	Outputs:
		val - (nbins, ) numpy array of the histogram weights
		centers - (nbins, ) numpy  array of the bin center values
		edges - (nbins+1, ) numpy array of the bin edges
	"""

    med, mad = estimateDistributionParameters(image)

    # Create bins. +/- 3*mad
    if minRange and 2*nsigma*mad < minRange:
    	bins = np.arange(np.floor(med - minRange/2), np.ceil(med + minRange/2))
    else:
	    bins = np.arange(np.floor(med - nsigma * mad), np.ceil(med + nsigma * mad))
    val, edges = np.histogram(image, bins=bins)
    centers = edges[:-1] + np.diff(edges)[0] / 2

    return val, centers, edges

def estimateDistributionParameters(image, ):
    """
    Utility function to compute the median and mad of an image used to build the histograms. Includes a few logical checks
    regarding saturated (0 ADU) pixels
    Inputs:
        image - (n x m x k) ndarray containing pixel values. This gets flattened so the original shape does not get retained
    Outputs:
        median - double of the median value of the pixels excluding zeros (saturated)
        mad - double of the median absolute deviation of the pixels excluding zeros
    """


    if np.all( image <= 0 ):
        # If all pixels are saturated, median = 0, mad = 1
        return 0, 1
    else:
        med = np.median(image[image > 0])
        mad = scipy.stats.median_absolute_deviation(image[image > 0], axis=None)

    return med, mad

if __name__ == "__main__":

    filename = "../Img_10.fits"

    _, data = readFits.read(filename)

    # Test entopy slope
    # slope, err, entropy = imageEntropySlope(data[:,:,:25])
    # print(convertValErrToString((slope, err)))

    # plt.plot(entropy, "*k")

    # Test noise fit
    # print("Test Image noise")
    # print(computeImageNoise(data[:,:,:-1], maxNskips=25))

    # Test cluster variance distribution
    print(computeImageTailRatio(data[:,:,:-1]))
    print(computeImageTailRatio(np.random.normal(10000, 60, data.shape)))

    # sigma = computeSkImageNoise(data[:, :, -1], nMovingAverage=5)
    # plt.show()
