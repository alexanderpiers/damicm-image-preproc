import argparse
import regex as re
import sys
import os
import inspect

sys.path.append("AutoAnalysis")


class analysisOutput(object):
    """Class that contains the information on the output results of the image analysis"""
    def __init__(self, filename, noise=-1, atest=-1, header=[]):
        super(analysisOutput, self).__init__()

        self.noise = noise
        self.atest = atest

        self.filename = filename
        self.header = list(header)
        self.headerString = ""

        # Create a header string for writing out to file
        self.headerString = "filenames\t"
        for s in self.header:
            self.headerString += str(s) + "\t"
        self.headerString += "\n"

    def __str__(self):
        """ Prints information regarding the analysis of an image """
        # Define output string
        outstr = "Analysis of " + self.filename + "\n"

        # Get all the user defined variables that contain information on the image analysis
        classAttributesAndMethods = inspect.getmembers(self, lambda a : not(inspect.isroutine(a)))
        classAnalysisAttributes = [a for a in classAttributesAndMethods if not(a[0].startswith("__") and a[0].endswith("__") or("header" in a[0]) or (a[0] == "filename"))]
        
        # Print the current value of all the analysis variables
        for analysisVar in classAnalysisAttributes:
            outstr += "\t"
            outstr += analysisVar[0] + ": " + str(analysisVar[1])
            outstr += "\n"
        return outstr

    def getTableString(self):
        """ 
            Returns a string of the class values depending on what values we want to return specified in the header attribute 
            For printing values in tabulated format.
        """
        outstr = self.filename + "\t"
        for outVar in self.header:
            outstr += str(self.__getattribute__(outVar))
            outstr += "\t"
        outstr += "\n"
        return outstr


def main(argv):
    """
	Command line entry into damic-m image preprocessing code. See printHelp() for use details
		-f - input filenames. Takes a single string and process any files that match the string
		-o - output file destination fo the results of the processing
	"""

    # Define the parser object
    parser = argparse.ArgumentParser(
        description="Command line interface for damic-m image preprocessing."
    )

    # Add arguments
    parser.add_argument(
        "-f",
        "--filenames",
        nargs="*",
        default=["Img_*.fits"],
        help="Processes all filese that match the string.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="img_preproc_out.txt",
        help="Output file for preprocessing information.",
    )
    parser.add_argument(
    	"-d", 
    	"--directory", 
    	default=os.getcwd(),
    	help="Directory to search for files in ")


    commandArgs = parser.parse_args()
    print(commandArgs)

    # Get a list of files in the directory
    filepath = commandArgs.directory
    outfile = commandArgs.output
    files = [
        f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))
    ]

    files2Process = []

    # Iterate over all the file strings passed to find matches
    for fn in commandArgs.filenames:
		# Define regex matching
    	regexPattern = re.compile(fn)
    	files2Process.extend(list(filter(regexPattern.match, files)))
    files2Process.sort()

    # Check what images have already been analyzed and written to file so we do not need to recompute
    try:
        with open(outfile) as of:
            print("Reading existing processed images")
            existingImgFiles =  [line.split()[0] for line in of]
    except FileNotFoundError:
        existingImgFiles = []

    # Process images with functions necessary to characterize the quality of images
    processedImgFiles = []
    processHeader = []
    print("Processing: ")
    for fp in files2Process:
        # Check to make sure the file has not already been processed
        if not(fp in existingImgFiles):
            print("\t" + os.path.join(filepath, fp))
            processedImgFiles.append(analysisOutput(fp, header=processHeader))


    # Appends new images to the output file or creates new file if doesn't exist
    if os.path.isfile(outfile):
        of = open(outfile, "a")
    else:
        of = open(outfile, "w+")
        try:
            of.write(processedImgFiles[0].headerString)
        except IndexError:
            pass

    # Write all the processed data
    for processedImg in processedImgFiles:
        of.write(processedImg.getTableString())
    of.close()


    return

if __name__ == "__main__":
    main(sys.argv[1:])
