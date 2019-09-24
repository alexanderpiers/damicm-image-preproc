import numpy as np
import scipy as scp

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