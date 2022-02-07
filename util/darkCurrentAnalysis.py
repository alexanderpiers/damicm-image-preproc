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
colOffset = 10

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

def plotPixelSpectrum(img, reverse=False, gain=-1000):

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Initial Poisson gaus fit
    fit = pgf.computeGausPoissDist(img, aduConversion=gain, npoisson=40, darkCurrent=-1)
    params = fit.params
    par = pgf.paramsToList(params)
    print(par)

    # Perform image mask

    # radius = 10
    # reversedImage = (img.image - params["offset"].value)
    # imageMask = tk.mask(reversedImage, 4 * params["ADU"].value, yradius, xradius)
    # imageMask.astype("int")

    # # img.image[np.logical_not(imageMask)] = 1e6

    # figimage, aximage = plt.subplots(1, 1)
    # # cax = aximage.imshow(reversedImage, aspect="auto", vmin=-3*np.std(reversedImage), vmax=3*np.std(reversedImage))
    # cax = aximage.imshow(imageMask, aspect="auto")
    # # figimage.colorbar(cax)

    # # Post masking fit
    # maskimg = DamicImage.DamicImage(img.image[imageMask].flatten(), bw=imagebw, reverse=reverse)
    # fit = pgf.computeGausPoissDist(maskimg, aduConversion=-1000, npoisson=40, darkCurrent=-1)
    # params = fit.params
    # par = pgf.paramsToList(params)
    # print(par)

    # Convert x axis to electrons
    xelectron = convertADUtoElectrons(img.centers, params)

    # Pixel Distribution
    ax.errorbar(xelectron, img.hpix, yerr=np.sqrt(img.hpix), fmt="ok", markersize=3, alpha=0.7)


    # Plot fit results
    x = np.linspace(xelectron[0], xelectron[-1], 2000)
    x = np.linspace(img.edges[0], img.edges[-1], 2000)
    fPoisGaus = pgf.fGausPoisson(x, *par)
    ax.plot(convertADUtoElectrons(x, params), fPoisGaus, "--r", linewidth=3)


    ax.set_xlabel("Pixel Value (e-)", fontsize=16)
    ax.set_ylabel("Counts / %.2f e-"%( imagebw / params["ADU"].value ), fontsize=16)
    ax.set_yscale("log")
    ax.set_ylim(0.05, params["N"].value / 2)
    ax.set_xlim(xelectron[img.hpix > 0][0] - 1, xelectron[-1])
    # ax.set_xlim(-1, 5)
    ax.legend(["$\sigma$=%.2f e- \n $\lambda$=%.2f e- / pix / exposure"%(params["sigma"].value / params["ADU"].value, params["lamb"].value)], fontsize=16, frameon=False)



    return fig, ax, fit

def computeMeanDarkCurrent(data):

    img = DamicImage.DamicImage(np.mean(data[:, :, skipOffset:-1], axis=-1), )

    fit = pgf.computeGausPoissDist(img, aduConversion=-7, npoisson=40, darkCurrent=-1)

    return fit.params["lamb"].value



def plotDarkCurrentRows(fullImage, rowbinswidth=1, reverse=True, mask=False, plotall=False, ax=None, color=colors[0],  readouttime=1, imageBinning=1, gain=-1000):

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    nrows = fullImage.image.shape[0]

    if plotall:
        _, axspec = plt.subplots(nrows // rowbinswidth, 1, sharex=True, figsize=(8, 16))

    rows = []
    darkcurrent = []
    darkcurrentErr = []

    fit = pgf.computeGausPoissDist(fullImage, aduConversion=gain, npoisson=20, darkCurrent=-1)
    params = fit.params

    if mask:


        reversedImage = (fullImage.image - params["offset"].value) *(1, -1)[reverse]
        imageMask = tk.mask(reversedImage, 5 * params["ADU"].value, yradius, xradius)
        imageMask.astype("int")


    for i in range(nrows // rowbinswidth):

        data = fullImage.image[i * rowbinswidth : (i+1) * rowbinswidth, :]

        if mask:
            data = data[imageMask[i * rowbinswidth : (i+1) * rowbinswidth, :]].flatten()

        img = DamicImage.DamicImage(data, bw=imagebw, reverse=reverse, minRange=minimumRange)

        fit = pgf.computeGausPoissDist(img, aduConversion=-params["ADU"].value, npoisson=20, darkCurrent=-params["lamb"].value, offset=-1, sigma=-params["sigma"].value)
        rows.append( (i+1) * rowbinswidth )
        darkcurrent.append(fit.params["lamb"].value)
        darkcurrentErr.append(fit.params["lamb"].stderr)

        if plotall:

            axspec[i].errorbar(img.centers, img.hpix, yerr=np.sqrt(img.hpix), fmt="ok", markersize=2)
            x = np.linspace(img.edges[0], img.edges[-1], 2000)
            axspec[i].plot(x, pgf.fGausPoisson(x, *pgf.paramsToList(fit.params)), "--r", linewidth=3)
            axspec[i].set_yscale("log")
            axspec[i].set_ylim(0.5, fit.params["N"].value / 2)
            axspec[i].set_xlim(img.centers[img.hpix > 0][0] - 1, img.centers[-1])



    print(np.array(darkcurrent))
    darkcurrent = np.array(darkcurrent) / imageBinning
    print(darkcurrentErr)

    try:
        darkcurrentErr = np.array(darkcurrentErr) / imageBinning
        ax.errorbar(rows, darkcurrent, yerr=darkcurrentErr, fmt="o", color=color)
    except:
        ax.plot(rows, darkcurrent, "o", color=color)
    ax.set_xlabel("Row Number", fontsize=16)
    ax.set_ylabel("Mean Charge (e- / pixel)", fontsize=16)

    flinear = lambda x, *p: p[0]*x + p[1]
    p0 = [(darkcurrent[-1] - darkcurrent[0]) / rows[-1], darkcurrent[0]]
    popt, pcov = curve_fit(flinear, rows, darkcurrent,  sigma=darkcurrentErr, p0=p0)

    # compute dark current in e- / pix / day
    darkCurrent = popt[0] * nrows / readouttime * 3600 * 24
    darkCurrentError = np.sqrt(pcov[0,0]) * nrows / readouttime * 3600 * 24
    ax.plot(rows, flinear(np.array(rows), *popt), "--r", linewidth=3, label="Dark Current: {:.2g} e- / pix / day".format(darkCurrent))
    ax.legend(fontsize=16)
    print("Lambda = {:.2g} +/- {:.1g} e- / pix / day".format(darkCurrent, darkCurrentError))
    return ax


if __name__ == '__main__':
    
    # argument parser
    parser = argparse.ArgumentParser(description="Plotting dark current of an image")

    parser.add_argument("-f", "--filename", nargs="*", help="Skipper FITS file to plot. Allows pattern matching")
    parser.add_argument("-r", "--reverse", action="store_true", help="Parity flip of the histogram")
    parser.add_argument("-p", "--parameter", help="Perform analysis on a given parameter scan")
    parser.add_argument("-m", "--mask", action="store_true", help="Mask clusters in image")
    parser.add_argument("-l", "--lta", action="store_true", help="LTA data")
    parser.add_argument("-e", "--extension", type=int, default=2)
    parser.add_argument("-g", "--gain", type=float, default=-1000)
    args = parser.parse_args()



    # distribute command line args to variables
    filename = args.filename
    reverse = args.reverse
    parameterScan = args.parameter
    mask = args.mask
    uselta = args.lta
    ext = args.extension
    gain = args.gain


    # Split filename into directory and filename
    # directory, filename = os.path.sdataplit(filename)
    # files = [ f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) ]

    # filepattern = re.compile(filename)
    # filesmatched = list(filter(filepattern.match, files))

    print(filename)


    if parameterScan == None:
        # datafile = os.path.join(directory, filesmatched[0])

        datafile = filename[0]


        # read data
        if uselta:
            header, data = readFits.readLTA(datafile)
            header = header[ext]
            print(header)
            nskips = int(header["NSAMP"])
            totalReadoutTime, totalBinning = getDCParametersFromHeader(header)
            data = readFits.reshapeLTAData(data[ext], int(header["NROW"]), int(header["NCOL"]), nskips)
            print(totalReadoutTime)
            print(totalBinning)
        else:
            header, data = readFits.read(datafile)
            data = data[:,:3000,:]


        # plot overall spectrum
        meanImage = np.mean(data[rowOffset:400, colOffset:308, skipOffset:], axis=-1)
        
        medianRowValue = np.median(meanImage, axis=-1)
        meanImage -= np.reshape( np.tile( medianRowValue, meanImage.shape[-1]), meanImage.shape, order="F")
        fullImage = DamicImage.DamicImage(meanImage, bw=imagebw, reverse=reverse, minRange=minimumRange)
        fig, ax, fit = plotPixelSpectrum(fullImage, reverse=reverse, gain=gain)


        # row dark current
        axrow = plotDarkCurrentRows(fullImage, plotall=True, mask=mask, reverse=reverse, rowbinswidth=75, readouttime=totalReadoutTime, imageBinning=totalBinning, gain=gain)

    else:
        legend = []
        darkcurrent = []
        paramscan = []

        fig, (ax, axp) = plt.subplots(2, 1, figsize=(12, 8))
        for i, file in enumerate(filename):

            # datafile = os.path.join(directory, file)
            datafile = file
            header, data = readFits.read(datafile)
            data = data[:,:3000,:]

            darkcurrent.append(computeMeanDarkCurrent(data))
            paramscan.append(header[parameterScan])

            legend.append(header[parameterScan])


            # if i == 0:
            #     ax = plotDarkCurrentRows(data, color=colors[i])
            # else:
            ax = plotDarkCurrentRows(data, ax=ax, color=colors[i])

        ax.legend(legend, fontsize=16, frameon=False)
        ax.set_xlabel("Row Number", fontsize=16)
        ax.set_ylabel("Dark Current (e- / pixel / row)", fontsize=16)
        ax.set_ylim(0, 0.5)

        # figp, axp = plt.subplots(1, 1, figsize=(12, 8))
        axp.plot(paramscan, darkcurrent, "o", color=colors[0])
        axp.set_xlabel(parameterScan, fontsize=16)
        axp.set_ylabel("Dark Current (e- / pixel / exposure)", fontsize=16)

    plt.show()
