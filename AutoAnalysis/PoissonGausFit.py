import numpy as np
import scipy.stats
import scipy.optimize as optimize
import readFits
import matplotlib.pyplot as plt
from scipy.special import factorial
import lmfit
import DamicImage



def computeGausPoissDist(damicImage, aduConversion=-1, npoisson=10):
    """
        Computes pixel distribution as a convolution of gaussian with poisson
    """


    # Set parameters to the fit
    params = lmfit.Parameters()
    params.add("sigma", value=damicImage.mad)
    params.add("lamb", value=0.5, min=0)
    params.add("offset", value=damicImage.med)
    if aduConversion > 0:
        params.add("ADU", value=aduConversion, vary=False)
    else:
        params.add("ADU", value=5)
    params.add("N", value=damicImage.image.size)
    params.add("npoisson", value=npoisson, vary=False)
    minimized = lmfit.minimize(lmfitGausPoisson, params, args=(damicImage.centers, damicImage.hpix))

    # Operations on the returned values to parse into a useful format
    return minimized

def fGausPoisson(x, *par):
    """
        Convolution of a gaussian and poisson.

        Inputs:
            x - double, value of function to be evaluated
            par - list of parameters
                par[0] - sigma, width of gaussians
                par[1] - lamb, mean of poisson process (lambda is python reserved keyword)
                par[2] - offset, shift of distribution relative to zero
                par[3] - a, electron to ADU conversion
                par[4] - N, amplitude of distribution (npixels)
                par[5] - npoiss, number of terms in the poisson process. This should be fixed

        Outputs:
            double, value of the function
    """
    sigma = par[0]
    lamb = par[1]
    offset = par[2]
    a = par[3]
    N = par[4]
    npoiss = par[5]

    y = 0
    for k in range(npoiss):
        y += ( lamb**k * np.exp(-lamb) / factorial(k) * np.exp( - (a*k - (x - offset))**2 / (2 * sigma**2)) )

    return y * N / np.sqrt(2 * np.pi * sigma**2)

def fCDFGausPoisson(x, *par):
    """ 
        Cumulative distribution function of a gaussian convolved with a poisson

        Inputs:
            x - double, value of function to be evaluated
            par - list of parameters
                par[0] - sigma, width of gaussians
                par[1] - lamb, mean of poisson process (lambda is python reserved keyword)
                par[2] - offset, shift of distribution relative to zero
                par[3] - a, electron to ADU conversion
                par[4] - N, amplitude of distribution (npixels)
                par[5] - npoiss, number of terms in the poisson process. This should be fixed

        Outputs:
            double, value of the cumulative distribution function
    """
    sigma = par[0]
    lamb = par[1]
    offset = par[2]
    a = par[3]
    N = par[4]
    npoiss = par[5]

    y = 0
    for k in range(npoiss):
        y += ( lamb**k * np.exp(-lamb) / factorial(k) * 0.5 * (1 + erf((x - offset - a*k) / (sigma * np.sqrt(2)))) )  

    return y 


def lmfitGausPoisson(param, x, data):
    """
    LMFIT function for a gaussian convolved with a poisson distribution
    """

    sigma = param["sigma"]
    lamb = param["lamb"]
    offset = param["offset"]
    a = param["ADU"]
    N = param["N"]
    npoiss = param["npoisson"]
    par = [sigma, lamb, offset, a, N, npoiss.value]

    model = fGausPoisson(x, *par)
    return (data-model)

def parseFitMinimum(fitmin):
    """
        Takes to fit minimum and parses it into a dictionary of useful parameters
    """

    params = fitmin.params
    output = {}
    output["sigma"]  = [ params["sigma"].value, params["sigma"].stderr ]
    output["lambda"] = [ params["lamb"].value,  params["lamb"].stderr  ]
    output["ADU"]    = [ params["ADU"].value,   params["ADU"].stderr   ]

    return output

def paramsToList(params):
    """
        Converts lmfit.params to a list. Only for poiss + gauss function
    """

    par = [ params["sigma"].value, params["lamb"].value, params["offset"].value, params["ADU"].value, params["N"].value, params["npoisson"].value]
    return par


if __name__ == "__main__":

    filename = "C:/Users/95286/Documents/damic_images/FS_Avg_Img_17.fits"
    # filename = "../Img_00.fits"

    header, data = readFits.read(filename)

    # Test datark current
    damicimage = DamicImage.DamicImage(data[:, :, -1], reverse=False, minRange=500)
    plt.figure()
    plt.hist(damicimage.centers, bins=damicimage.edges, weights=damicimage.hpix) # Plot histogram of data


    # Perform poisson gaus fit to data
    minres = computeGausPoissDist(damicimage, )
    params = minres.params
    print(lmfit.fit_report(minres))
    print(parseFitMinimum(minres))


    left = 49600
    right = 49800

    # Plot fit results
    par = paramsToList(params)
    x = np.linspace(damicimage.centers[0], damicimage.centers[-1], 2000)
    plt.plot(x, fGausPoisson(x, *par), "--r")
    plt.yscale("log")
    plt.ylim(0.01, params["N"])
    plt.xlim([left,right])


    # Masked Image
    maskedimage = DamicImage.MaskedImage(data[:,:,-1], reverse=False, minRange=500)
    plt.figure()
    plt.hist(maskedimage.centers, bins=maskedimage.edges, weights=maskedimage.hpix)

    #Perform poisson gaus fit to data
    minres = computeGausPoissDist(maskedimage, )
    params = minres.params
    print(lmfit.fit_report(minres))
    print(parseFitMinimum(minres))



    # Plot fit results
    par = paramsToList(params)
    x = np.linspace(maskedimage.centers[0], maskedimage.centers[-1], 2000)
    plt.plot(x, fGausPoisson(x, *par), "--r")
    plt.yscale("log")
    plt.ylim(0.01, params["N"])
    plt.xlim([left,right])

    # Goodness of Fit

    def PGD(x,*par):
        return fGausPoisson(x, *par)/par[4]

    D, pval = scipy.stats.kstest(maskedimage.hpix/par[4], PGD, args=par, N=par[4])
    print("D = ", D)
    print("p-value = ", pval)


    plt.show()