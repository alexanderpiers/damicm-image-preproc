import argparse
import regex as re
import sys
import os


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
    print("Processing: ")
    for fp in files2Process:
        print("\t" + os.path.join(filepath, fp))

    return

if __name__ == "__main__":
    main(sys.argv[1:])
