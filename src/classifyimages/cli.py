#-----------------------------------------------------------------------------------------------------------------------
# Module:  cli.py
# Project: classifyimages
# Version: 0.1.0
# Date:    2023-03-01
# Author:  Rohin Gosling
#
# Description:
#
#   Define the command-line interface without importing image or model libraries.
#-----------------------------------------------------------------------------------------------------------------------

from argparse import ArgumentParser, ArgumentTypeError, ArgumentDefaultsHelpFormatter, SUPPRESS
from math     import isfinite
from pathlib  import Path

from classifyimages.limits import MAX_COLLECTION_IMAGES


#-----------------------------------------------------------------------------------------------------------------------
# Function: parse_alpha
#
# Description:
#
#   Accept a finite content contribution in the specified interval [0, 1].
#
# Arguments:
#
#   value : Text supplied for the command-line option.
#
# Returns:
#
#   Validated finite content contribution in [0, 1].
#-----------------------------------------------------------------------------------------------------------------------

def parse_alpha ( value: str ) -> float:

    # Accept a finite content contribution in the specified interval [0, 1].

    try:
        alpha = float ( value )
    except ValueError as error:
        raise ArgumentTypeError ( "alpha must be a number in [0, 1]" ) from error
    if not isfinite ( alpha ) or not 0.0 <= alpha <= 1.0:
        raise ArgumentTypeError ( "alpha must be a finite number in [0, 1]" )

    # Return the validated contribution.

    return alpha


#-----------------------------------------------------------------------------------------------------------------------
# Function: parse_color_weight
#
# Description:
#
#   Accept a finite color contribution expressed as a percentage from 0 through 100.
#
# Arguments:
#
#   value : Text supplied for the command-line option.
#
# Returns:
#
#   Validated finite color contribution in [0, 100].
#-----------------------------------------------------------------------------------------------------------------------

def parse_color_weight ( value: str ) -> float:

    # Accept a finite color contribution expressed as a percentage from 0 through 100.

    try:
        color_weight = float ( value )
    except ValueError as error:
        raise ArgumentTypeError ( "color weight must be a percentage in [0, 100]" ) from error
    if not isfinite ( color_weight ) or not 0.0 <= color_weight <= 100.0:
        raise ArgumentTypeError ( "color weight must be a finite percentage in [0, 100]" )

    # Return the validated percentage; the entry point converts it to the content contribution.

    return color_weight

#-----------------------------------------------------------------------------------------------------------------------
# Function: parse_threshold
#
# Description:
#
#   Accept a finite, nonnegative cosine distance cut.
#
# Arguments:
#
#   value : Text supplied for the command-line option.
#
# Returns:
#
#   Validated finite nonnegative cosine distance cut.
#-----------------------------------------------------------------------------------------------------------------------

def parse_threshold ( value: str ) -> float:

    # Accept a finite, nonnegative cosine distance cut.

    try:
        threshold = float ( value )
    except ValueError as error:
        raise ArgumentTypeError ( "threshold must be a nonnegative number" ) from error
    if not isfinite ( threshold ) or threshold < 0.0:
        raise ArgumentTypeError ( "threshold must be a finite nonnegative number" )

    # Return the validated distance.

    return threshold


#-----------------------------------------------------------------------------------------------------------------------
# Function: create_parser
#
# Description:
#
#   Build the public argument contract without loading the image processing libraries.
#
# Arguments:
#
#   None.
#
# Returns:
#
#   Configured command-line argument parser.
#-----------------------------------------------------------------------------------------------------------------------

def create_parser () -> ArgumentParser:

    # Build the public argument contract without loading the image processing libraries.

    parser = ArgumentParser (
        prog        = "classifyimages",
        description = "Group images by visual similarity and write full-resolution RGB PNGs.",
        epilog = (
            f"Maximum {MAX_COLLECTION_IMAGES:,} supported-extension files per run, including unreadable files. "
            "Split larger collections into smaller folders. "
            "Existing PNG targets and manifests are never overwritten. "
            "Destination and manifest must be outside the source folder. Keep source files unchanged while processing. Exit codes: 0 success, 2 usage/collision/empty input, 3 model failure, 4 write failure."
        ),
        formatter_class = ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument ( "-s", "--source", type = Path, required = True, help = "folder of input images" )
    parser.add_argument ( "-d", "--dest", type = Path, required = True, help = "folder for classified PNG output" )
    parser.add_argument ( "--threshold", type = parse_threshold, default = 0.35, help = "average-linkage cosine distance cut" )
    weighting = parser.add_mutually_exclusive_group ()
    weighting.add_argument (
        "--color-weight", type = parse_color_weight, default = 20.0, metavar = "PERCENT",
        help                   = "color contribution from 0 to 100 percent; 0 uses content only, 100 uses color only",
    )
    weighting.add_argument (
        "--alpha", type = parse_alpha, default = SUPPRESS,
        help            = "legacy content contribution in [0, 1]; equivalent to color weight 100 * (1 - alpha)",
    )
    parser.add_argument ( "--recursive", action = "store_true", help = "descend into source subfolders" )
    parser.add_argument ( "--manifest", type = Path, help = "CSV map path; defaults to <dest>/manifest.csv" )
    parser.add_argument ( "--model", type = Path, help = "override the bundled ONNX model path" )

    # Return the parser without touching the filesystem.

    return parser
