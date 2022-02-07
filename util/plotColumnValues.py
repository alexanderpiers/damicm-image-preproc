import numpy as np
import argparse
import sys
import os
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import curve_fit

sys.path.append("../AutoAnalysis")

import readFits

def fgausCDF(x, N, mu, sigma):

    return N*scipy.stats.norm.cdf(x, mu, sigma)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename")
    parser.add_argument("-e", "--ext", default=4, type=int)
    parser.add_argument("-x", "--exposure", default=3600, type=float)
    parser.add_argument("-g", "--gain", default=1000, type=float)
    args = parser.parse_args()

    infile = args.filename
    extension = args.ext
    exposure = args.exposure
    gain = args.gain

    header, data = readFits.readLTA(infile)

    header = header[extension]
    nrow = int(header["NROW"])
    ncol = int(header["NCOL"])
    nskip = int(header["NSAMP"])

    dataU = readFits.reshapeLTAData(data[4], nrow, ncol, nskip)
    dataL = readFits.reshapeLTAData(data[2], nrow, ncol, nskip)


    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    print(dataU.shape)
    colValueU = np.median(dataU[:, :, 0], axis=0)
    colValueL = np.median(dataL[:, :, 0], axis=0)

    print(colValueL.shape)
    ax.plot(colValueU - np.median(colValueU), "ok", alpha=0.3, label="U2")
    ax.plot(colValueL - np.median(colValueL), "vr", alpha=0.3, label="L2")

    ax.set_xlabel("Column Number", fontsize=16)
    ax.set_ylabel("Median Subtracted Column Value (ADU)", fontsize=16)

    # perform fit to the overscan
    muInit = 3080
    sigmaInit = 1
    Ninit = np.mean(colValueU[-100:]) - np.mean(colValueU[2000:3000])
    column = np.arange(colValueU.size)
    print([Ninit, muInit, sigmaInit])
    try:
        poptU, covU = curve_fit(fgausCDF, column[1500:], colValueU[1500:]-np.median(colValueU), p0=(Ninit, muInit, sigmaInit))
        poptL, covL = curve_fit(fgausCDF, column[1500:], colValueL[1500:]-np.median(colValueL), p0=(Ninit, muInit, sigmaInit))
        ax.plot(column, fgausCDF(column, *poptU), "--k", linewidth=3, alpha=0.5, label="Dark Rate={:.2f} e- / pix / day".format(-poptU[0] / gain / exposure * 3600 * 24))
        ax.plot(column, fgausCDF(column, *poptL), "--r", linewidth=3, alpha=0.5, label="Dark Rate={:.2f} e- / pix / day".format(-poptL[0] / gain / exposure * 3600 * 24))
        print(poptU)
        print(np.sqrt(covU))
    except Exception as e:
        print(e)
        pass
    ax.legend(fontsize=16)
    plt.show()
