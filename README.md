# damicm-image-preproc

Information on the image preprocessing and intial characterization of DAMIC-M skipper images. 

## Requirements

This code uses Python3 and the packages listed in requirements.txt to run. It is recommended that you use a virtualenv and run `pip install -r requirements.txt` to get all the necessary packages.

## Usage

This code is intended to process images stored in fits files to make a quick evaluation on the quality of the image. The usage is:

```python image_preproc.py```

By default, the code searchs for any files matching "Img_[0-9]+.fits" in the current working directory, performs a series of desired computations, and writes the result in a tabulated form in the file "image_preproc_out.txt".

Additionally, the following command line arguments can be used:

```
-f --filenames <filenames>
-o --output <outfilename>
-d --directory <dir>
-a --all
```

The arguments for `-f` can be a list of filenames (also supports wildcard matching) or a regex pattern, and all files in the directory that match the pattern will be processed. `-d` is the directory to search for files to process. By default, only files where the processed results are not in the `<outfilename>` will be processed; to process all files that match, use the `-a` flag.

Use `python image_preproc.py -h` for help.
