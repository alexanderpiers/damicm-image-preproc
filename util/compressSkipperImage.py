import numpy as np
import argparse
import sys
from astropy.io import fits
import os
sys.path.append("/home/b059ante/Documents/software/damicm-image-preproc/AutoAnalysis")

import readFits

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename")
    parser.add_argument("-e", "--ext", default=4, type=int)
    parser.add_argument("-s", "--skoffset", default=1, type=int)
    parser.add_argument("-o", "--outdir", default="/home/b059ante/Documents/data/processed")

    args = parser.parse_args()

    infile = args.filename
    extension = args.ext
    skoffset = args.skoffset
    outdir = args.outdir

    # read file
    header, data = readFits.readLTA(infile)

    header = header[extension]
    nrow = int(header["NROW"])
    ncol = int(header["NCOL"])
    nskip = int(header["NSAMP"])

    data = readFits.reshapeLTAData(data[extension], nrow, ncol, nskip)

    compressedData = np.mean(data[:,:,skoffset:], axis=-1)

    # create the output filename
    print(os.path.split(infile))
    outfile = os.path.splitext(os.path.split(infile)[-1])[0] +"_ext_" + str(extension) + "_compr_" + str(skoffset) + "_" + str(nskip) +  ".fits"
    print(os.path.join(outdir, outfile))
    outfits = fits.writeto(os.path.join(outdir, outfile), compressedData, header=header)

    print("Compressed file!")
