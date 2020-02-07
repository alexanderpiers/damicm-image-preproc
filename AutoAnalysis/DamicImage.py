import numpy as np
import matplotlib.pyplot as plt
import readFits
import scipy.stats

class Image(object):
	"""docstring for Image"""
	def __init__(self, img, filename="", minRange=None):
		super(Image, self).__init__()
		self.image = img
		self.filename = filename

		# Compute usefule statistics on the image
		self.estimateDistributionParameters()
		self.histogramImage(minRange=minRange)



	def estimateDistributionParameters(self,):
	    """
	    Utility function to compute the median and mad of an image used to build the histograms. Includes a few logical checks
	    regarding saturated (0 ADU) pixels
	    Inputs:
	        image - (n x m x k) ndarray containing pixel values. This gets flattened so the original shape does not get retained
	    Outputs:
	        median - double of the median value of the pixels excluding zeros (saturated)
	        mad - double of the median absolute deviation of the pixels excluding zeros
	    """


	    if np.all( self.image <= 0 ):
	        # If all pixels are saturated, median = 0, mad = 1
	        self.med = 0
	        self.mad = 1
	    else:
	        self.med = np.median(self.image[self.image > 0])
	        self.mad = np.max([scipy.stats.median_absolute_deviation(self.image[self.image > 0], axis=None), 1])

	    return self.med, self.mad

	def histogramImage(self, nsigma=3, minRange=None):
	    """
		Creates a histogram of an image (or any data set) of a reasonable range with integer (ADU) spaced bins
		Inputs:
			nsigma - number of median absolute deviations around the median we histogram
			minRange - if provided this sets the minimum range (if nsigma range is less, this supercedes it)
		Outputs:
			val - (nbins, ) numpy array of the histogram weights
			centers - (nbins, ) numpy  array of the bin center values
			edges - (nbins+1, ) numpy array of the bin edges
		"""


	    # Create bins. +/- 3*mad
	    if minRange and 2*nsigma*self.mad < minRange:
	    	bins = np.arange(np.floor(self.med - minRange/2), np.ceil(self.med + minRange/2))
	    else:
		    bins = np.arange(np.floor(self.med - nsigma * self.mad), np.ceil(self.med + nsigma * self.mad))
	   
	    hpix, edges = np.histogram(self.image, bins=bins)
	    centers = edges[:-1] + np.diff(edges)[0] / 2

	    self.hpix, self.centers, self.edges = hpix, centers, edges

	    return hpix, centers, edges


	
		

class DamicImage(Image):
	"""docstring for SkipperImage"""
	def __init__(self, img, reverse=True, filename="", minRange=200):

		super(DamicImage, self).__init__(img, filename=filename, minRange=minRange)
		self.reverse = reverse

		if self.reverse: self.reverseHistogram();

	def reverseHistogram(self):

		# Shifts the histogram axis (centers and edges) to be centered around zero and flips the values
		self.hpix = np.flip(self.hpix)
		# self.centers = (self.centers - self.med)
		# self.edges = (self.edges - self.med)

		

if __name__ == '__main__':

	# Test to see if reversing image works
	imgname = "../Img_11.fits"

	header, data = readFits.read(imgname)

	normalImage = DamicImage(data[:,:,-1], False, "normalImage test")
	reverseImage = DamicImage(data[:,:,-1], True, "forward test")

	normalImage.histogramImage(minRange=80)
	reverseImage.histogramImage(minRange=80)
	reverseImage.reverseHistogram()

	fig, axs = plt.subplots(1, 2, figsize=(14,8))
	axs[0].hist(normalImage.centers, weights=normalImage.hpix, bins=normalImage.edges)
	axs[1].hist(reverseImage.centers, weights=reverseImage.hpix, bins=reverseImage.edges)
	plt.show()