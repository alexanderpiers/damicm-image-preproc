import numpy as np
import argparse
import scipy.stats
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.append("/home/b059ante/Documents/software/damicm-image-preproc/AutoAnalysis")

import readFits

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Plot image pixel distribution")
    parser.add_argument("-f", "--filename")
    parser.add_argument("-e", "--ext", default=4, type=int)

    args = parser.parse_args()

    filename = args.filename
    extension = args.ext

    # Read file
    header, data = readFits.readLTA(filename)

    header = header[extension]
    data = readFits.reshapeLTAData(data[extension], int(header["NROW"]), int(header["NCOL"]), int(header["NSAMP"]))
    data = data[2:,50:,:]
    print(data.shape)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    binsize = 50
    bins = np.arange(np.min(data), np.max(data), binsize)
    binc = bins[:-1] + np.diff(bins)[0]/2
    h, _ = np.histogram(data.flatten(), bins=bins)
    herr = np.sqrt(h)

    med = np.median(data, axis=None)
    mad = scipy.stats.median_abs_deviation(data, axis=None)

    ax.errorbar(binc, h, yerr=herr, fmt="ok", label=r"med={:.1f}, mad={:.1f}".format(med, mad))
    ax.set_xlabel("Pixel Values (ADU)", fontsize=16)
    ax.set_yscale("log")

    fgaus = lambda x, N, mu, sigma: N*scipy.stats.norm.pdf(x, mu, sigma)
    popt, _ = curve_fit(fgaus, binc, h, p0=[data.size, med, mad])
    ax.plot(binc, fgaus(binc, *popt), "--r", label="$\mu=${:.1f}, $\sigma=${:.1f}".format(*popt[1:]))
    ax.legend(fontsize=16)
    ax.set_ylim(1, 1.2*np.max(h))

    plt.show()


