import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import palettable
import argparse
import scipy.stats
import scipy.signal
from scipy.optimize import curve_fit
import regex as re
import datetime
sys.path.append("../AutoAnalysis")

import PoissonGausFit as pgf 
import readFits
import DamicImage
import lmfit
# from trackMasking import mask
import trackMasking as tk


# Defining some default values
skipOffset = 1
imagebw = 50
fixADU = 1
yradius = 10
xradius = 10
minimumRange=8000
rowOffset = 1
colOffset = 150

colors = palettable.cmocean.sequential.Thermal_6.mpl_colors

def getDCParametersFromHeader(header):
    
    starttime = datetime.datetime.strptime(header["DATESTART"], "%Y-%m-%dT%H:%M:%S")
    stoptime = datetime.datetime.strptime(header["DATEEND"], "%Y-%m-%dT%H:%M:%S")
    totaltime = (stoptime - starttime).total_seconds() - float(header["EXPOSURE"])# in seconds
    totalPixelBin = int(header["NBINROW"]) * int(header["NBINCOL"])

    return totaltime, totalPixelBin

def convertADUtoElectrons(x, params):

    offset = params["offset"].value
    conversion = params["ADU"].value

    return (x - offset) / conversion



def plotImageAndRowColDC(fullImage, bincols, binrows, gain=-1000, readouttime=1, imagebinning=1):


    fig, axs = plt.subplots(2, 2, gridspec_kw = {'width_ratios':[1,3], 'height_ratios':[1,3]}, figsize=(16,10))
    nrows, ncols = fullImage.image.shape

    rows = []
    cols = []
    darkcurrentRows = []
    darkcurrentErrRows = []

    darkcurrentRows = []
    darkcurrentErrRows = []
    
    # perform an initial fit to get good parameters
    fit = pgf.computeGausPoissDist(fullImage, aduConversion=gain, npoisson=20, darkCurrent=-1)
    params = fit.params


    for i in range(nrows // binrows):

        data = fullImage.image[i * binrows : (i+1) * binrows, :]


        img = DamicImage.DamicImage(data, bw=imagebw, reverse=False, minRange=minimumRange)

        fit = pgf.computeGausPoissDist(img, aduConversion=-params["ADU"].value, npoisson=20, darkCurrent=-params["lamb"].value, offset=-1, sigma=-params["sigma"].value)
        rows.append( (i+1) * rowbinswidth )
        darkcurrent.append(fit.params["lamb"].value)
        darkcurrentErr.append(fit.params["lamb"].stderr)


    # plot the image
    colorGradient = palettable.cmocean.sequential.Amp_20.mpl_colormap
    cax = axs[1][1].imshow(img.image, aspect="auto",  cmap=colorGradient,interpolation="none", vmin=fullImage.med-3*fullImage.mad, vmax=fullImage.med+3*fullImage.mad,)
    #fig.colorbar(cax, ax=ax[0])
    ax[1][1].set_xlabel("x [pixels]", fontsize=14)
    ax[1][1].set_ylabel("y [pixels]", fontsize=14)
    darkcurrent = np.array(darkcurrent) / imageBinning


    try:
        darkcurrentErr = np.array(darkcurrentErr) / imageBinning
        ax[1][0].errorbar(darkcurrent, rows, xerr=darkcurrentErr, fmt="o", color=color)
    except:
        ax[1][0].plot(darkcurrent, rows, "o", color=color)
    axs[1][0].invert_xaxis()
    axs[1][0].set_ylabel("Row Number", fontsize=16)
    axs[1][0].set_xlabel("Mean Charge (e- / pixel)", fontsize=16)

    flinear = lambda x, *p: p[0]*x + p[1]
    p0 = [(darkcurrent[-1] - darkcurrent[0]) / rows[-1], darkcurrent[0]]
    popt, pcov = curve_fit(flinear, rows, darkcurrent,  sigma=darkcurrentErr, p0=p0)

    # compute dark current in e- / pix / day
    darkCurrent = popt[0] * nrows / readouttime * 3600 * 24
    darkCurrentError = np.sqrt(pcov[0,0]) * nrows / readouttime * 3600 * 24
    #ax[1][0].plot(rows, flinear(np.array(rows), *popt), "--r", linewidth=3, label="Dark Current: {:.2g} e- / pix / day".format(darkCurrent))
    #ax.legend(fontsize=16)
    print("Lambda = {:.2g} +/- {:.1g} e- / pix / day".format(darkCurrent, darkCurrentError))
    return ax


if __name__ == '__main__':
    
    # argument parser
    parser = argparse.ArgumentParser(description="Plotting dark current of an image")

    parser.add_argument("-f", "--filename", help="Skipper FITS file to plot. Allows pattern matching")
    parser.add_argument("-l", "--lta", action="store_true", help="LTA data")
    parser.add_argument("-e", "--extension", type=int, default=2)
    parser.add_argument("-k", "--gain", type=float, default=-1000)
    parser.add_argument("-r", "--rows", type=int, default=10, help="Number of rows to bin")
    parser.add_argument("-c", "--columns", type=int, default=10, help="Number of columns to bin")
    parser.add_argument("--correction", action="store_true")
    args = parser.parse_args()



    # distribute command line args to variables
    filename = args.filename
    uselta = args.lta
    ext = args.extension
    gain = args.gain
    columnsGroup = args.columns
    rowsGroup = args.rows
    correction = args.correction



    # read data
    header, data = readFits.readLTA(filename)
    header = header[ext]
    nskips = int(header["NSAMP"])
    totalReadoutTime, totalBinning = getDCParametersFromHeader(header)
    data = readFits.reshapeLTAData(data[ext], int(header["NROW"]), int(header["NCOL"]), nskips)


    # plot overall spectrum
    meanImage = np.mean(data[rowOffset:400, colOffset:308, skipOffset:], axis=-1)
    if correction:       
        medianRowValue = np.median(meanImage, axis=-1)
        meanImage -= np.reshape( np.tile( medianRowValue, meanImage.shape[-1]), meanImage.shape, order="F")
    fullImage = DamicImage.DamicImage(meanImage, bw=imagebw, reverse=False, minRange=minimumRange)

    fig, axs = plotImageAndRowColDC(fullImage, columnsGroup, rowsGroup, gain=gain, readouttime=totalReadoutTime, imagebinning=totalBinning)

    plt.show()
