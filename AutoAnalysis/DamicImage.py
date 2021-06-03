import numpy as np
import matplotlib.pyplot as plt
import readFits
import scipy.stats
import trackMasking as tk


class Image(object):
    """
	Image Class

	Used to store 2D numpy (N x M) arrays along with some statistical properties of the image (med, mad, histogram) etc.
	Provides some utility so that these values do not need to be continously recomputed.

	Use:
		image = DamicImage(data, filename=<string>, minRange=<double>)
	"""

    def __init__(self, img, filename="", minRange=None, bw=1):
        super(Image, self).__init__()
        self.image = img
        self.filename = filename

        # Compute usefule statistics on the image
        self.estimateDistributionParameters()
        self.histogramImage(minRange=minRange, bw=bw)

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

        if np.all(self.image <= 0):
            # If all pixels are saturated, median = 0, mad = 1
            self.med = 0
            self.mad = 1
        else:
            self.med = np.median(self.image[self.image > 0])
            self.mad = np.max(
                [
                    scipy.stats.median_absolute_deviation(
                        self.image[self.image > 0], axis=None
                    ),
                    1,
                ]
            )

        return self.med, self.mad

    def histogramImage(self, nsigma=3, minRange=None, bw=1):
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
        if minRange and 2 * nsigma * self.mad < minRange:
            bins = np.arange(
                np.floor(self.med - minRange / 2), np.ceil(self.med + minRange / 2), bw
            )
        else:
            bins = np.arange(
                np.floor(self.med - nsigma * self.mad),
                np.ceil(self.med + nsigma * self.mad), bw
            )

        hpix, edges = np.histogram(self.image, bins=bins)
        centers = edges[:-1] + np.diff(edges)[0] / 2

        self.hpix, self.centers, self.edges = hpix, centers, edges

        return hpix, centers, edges

    def plotSpectrum(self, bins=None):
        """
            Plots the histgram of the image
        """

        fig, ax = plt.subplots(1, 1)

        if np.any(bins):
            ax.hist(self.image.flatten(), bins=bins)
        else:
            # Use default binning
            if not hasattr(self, "hpix"):
                self.histogramImage(minRange=500)

            ax.hist(self.centers, bins=self.edges, weights=self.hpix)


        ax.set_title("Image Spectrum", fontsize=18)
        ax.set_xlabel("Pixel Value [ADU]", fontsize=14)

        return fig, ax

class DamicImage(Image):
    """
		DamicImage Class

		Extended utility to the image class to allow both "raw" and "processed average" images to be used by the other analysis functions.
		Depending on what processing as been done, increasing number of electrons may increase (normal) or decrease (reverse) the pixel value,
		so the reverse flag allows conversion from reverse into normal. All histograms should be in the "normal" schema for further procccessing

		Use:
			image = DamicImage(data, reverse=<bool>, filename=<string>, minRange=<double>)
	"""

    def __init__(self, img, reverse=True, filename="", minRange=200, bw=1):

        super(DamicImage, self).__init__(img, filename=filename, minRange=minRange, bw=bw)
        self.reverse = reverse

        if self.reverse:
            self.reverseHistogram()

    def reverseHistogram(self):

        # Shifts the histogram axis (centers and edges) to be centered around zero and flips the values
        self.hpix = np.flip(self.hpix)
        # self.centers = (self.centers - self.med)
        # self.edges = (self.edges - self.med)


class MaskedImage(DamicImage):
    """
        Accept the same parameters as DamicImage class and inherit all
        the functionalities. Create a mask that remove all the tracks in raw image

		Use:
			image = MaskedImage(data, reverse=<bool>, filename=<string>, minRange=<double>)

    """
    def __init__(self, img, reverse=True, filename="", minRange=200, bw=1, maskThreshold=16000, maskRadiusX=1, maskRadiusY=1):
        super(MaskedImage, self).__init__(img, reverse, filename, minRange, bw)
        self.mask = []
        self.image = self.mask(maskThreshold, maskRadiusX, maskRadiusY)
        super(MaskedImage,self).__init__(self.image, reverse, filename, minRange, bw)


    def mask(self, threshold, radiusx, radiusy):
  		# Create a mask and remove all the pixels around identified tracks
  	    # Ouputs:

    	self.mask = tk.mask(self.image, threshold, radiusx, radiusy)
    	maskedImage = self.image[mask].flatten()

   
if __name__ == "__main__":

    # Test to see if reversing image works
    imgname = "../Img_11.fits"

    header, data = readFits.read(imgname)

    normalImage = DamicImage(data[:, :, -1], False, "normalImage test")
    reverseImage = DamicImage(data[:, :, -1], True, "forward test")

    normalImage.histogramImage(minRange=80)
    reverseImage.histogramImage(minRange=80)
    reverseImage.reverseHistogram()

    fig, axs = plt.subplots(1, 2, figsize=(14, 8))
    axs[0].hist(normalImage.centers, weights=normalImage.hpix, bins=normalImage.edges)
    axs[1].hist(
        reverseImage.centers, weights=reverseImage.hpix, bins=reverseImage.edges
    )




    fig, axs = plt.subplots(1, 2, figsize=(14, 8))
    axs[0].hist(normalImage.centers, weights=normalImage.hpix, bins=normalImage.edges)
    axs[1].hist(
        reverseImage.centers, weights=reverseImage.mhpix, bins=reverseImage.edges
    )



    plt.show()
