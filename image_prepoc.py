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
        default="Img_*.fits",
        help="Processes all filese that match the string.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="img_preproc_out.txt",
        help="Output file for preprocessing information.",
    )

    commandArgs = parser.parse_args()

    print(commandArgs)

    # Takes the directory from --filename (or uses current dir if none is provided) to get all files in dir
    filepath, filepattern = os.split(commandArgs.filenames)
    if not filepath:
        filepath = os.getcwd()

    # Get a list of files that match with --filenames
    files = [
        f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))
    ]

    # Define regex matching
    regexPattern = re.compile(filepattern)
    files2Process = list(filter(regexPattern.match, files))

    for fp in files2Process:
        print(os.path.join(filepath, fp))

    return

if __name__ == "__main__":
    main(sys.argv[1:])
