#-----------------------------------------------------------------------------------------------------------------------
# Program: classifyimages
# Project: classifyimages
# Version: 1.0.0
# Date:    2023-03-01
# Author:  Rohin Gosling
#
# Description:
#
#   Shared entry point for the module invocation and installed console command.
#
# Usage:
#
#   python -m classifyimages -s C:\images -d C:\classified
#   imageclassifier.exe -s C:\images -d C:\classified
#-----------------------------------------------------------------------------------------------------------------------

from classifyimages.cli import create_parser


#-----------------------------------------------------------------------------------------------------------------------
# Function: _decimal
#
# Description:
#
#   Preserve user-specified precision while showing two decimal places for the default alpha.
#
# Arguments:
#
#   value : Numeric setting whose displayed precision must be retained.
#
# Returns:
#
#   Numeric text with at least two decimal places when needed for the default alpha.
#-----------------------------------------------------------------------------------------------------------------------

def _decimal ( value: float ) -> str:

    # Preserve user-specified precision while showing two decimal places for the default alpha.

    formatted = format ( round ( value, 10 ), ".10g" )
    if "." not in formatted and "e" not in formatted:

        # Return data to caller.

        return formatted + ".00"
    if formatted.endswith ( "." ) or len ( formatted.split ( "." ) [ -1 ] ) == 1:

        # Return data to caller.

        return formatted + "0"

    # Return data to caller.

    return formatted


#-----------------------------------------------------------------------------------------------------------------------
# Function: _print_banner
#
# Description:
#
#   Show the effective settings in the order used by the processing stages.
#
# Arguments:
#
#   options : Parsed command-line options.
#   alpha   : Content contribution in [0, 1]; color contributes 1 - alpha.
#   output  : Text stream receiving settings, progress, or summary output.
#
# Returns:
#
#   None.
#-----------------------------------------------------------------------------------------------------------------------

def _print_banner ( options, alpha: float, output ) -> None:

    # Show the effective settings in the order used by the processing stages.

    from classifyimages import __version__

    #-------------------------------------------------------------------------------------------------------------------
    # Function: setting
    #
    # Description:
    #
    #   Print one aligned setting row to the selected output stream.
    #
    # Arguments:
    #
    #   label : Setting or progress-stage label to display.
    #   value : Formatted setting value.
    #
    # Returns:
    #
    #   None.
    #-------------------------------------------------------------------------------------------------------------------

    def setting ( label: str, value: str ) -> None:

        # Print one aligned setting row to the selected output stream.

        print ( f"- {label + ':':<25} {value}", file = output )

    print ( file = output )
    print ( f"Image Classifier (Version {__version__})", file = output )
    print ( file = output )
    setting ( "Source", str ( options.source ) )
    setting ( "Destination", str ( options.dest ) )
    setting ( "Threshold", format ( options.threshold, "g" ) )
    setting ( "Feature Weight - Content", format ( round ( alpha * 100, 10 ), "g" ) )
    setting ( "Feature Weight - Color", format ( round ( ( 1.0 - alpha ) * 100, 10 ), "g" ) )
    setting ( "Alpha", _decimal ( alpha ) )
    setting ( "Recursive", "On" if options.recursive else "Off" )
    setting ( "Manifest", str ( options.manifest if options.manifest is not None else options.dest / "manifest.csv" ) )
    setting ( "Model", f"ONNX ({options.model})" if options.model is not None else "ONNX" )
    print ( file = output )
    output.flush ()


#-----------------------------------------------------------------------------------------------------------------------
# Function: main
#
# Description:
#
#   Keep help lightweight and map processing errors to the public exit-code contract.
#
# Arguments:
#
#   arguments : Command-line arguments, or None to read the process arguments.
#
# Returns:
#
#   Exit code 0 for success, 2 for usage, 3 for model errors, or 4 for write failures.
#-----------------------------------------------------------------------------------------------------------------------

def main ( arguments: list [ str ] | None = None ) -> int:

    # Keep help lightweight and map processing errors to the public exit-code contract.

    parser  = create_parser ()
    options = parser.parse_args ( arguments )

    # Delay image and inference imports until argument parsing has handled help and usage errors.

    from importlib import import_module
    from logging   import Formatter, Handler, WARNING, getLogger

    from classifyimages.cluster  import ClusteringError
    from classifyimages.features import ModelError
    from classifyimages.pipeline import run
    from classifyimages.writeout import UsageError, WriteError

    runtime       = import_module ( "sys" )
    logger        = getLogger ( "classifyimages" )
    alpha         = getattr ( options, "alpha", 1.0 - options.color_weight / 100.0 )
    warning_lines = []

    #-------------------------------------------------------------------------------------------------------------------
    # Class: BufferedWarnings
    #
    # Description:
    #
    #   Collect warning messages until processing progress has finished.
    #-------------------------------------------------------------------------------------------------------------------

    class BufferedWarnings ( Handler ):

        #---------------------------------------------------------------------------------------------------------------
        # Function: emit
        #
        # Description:
        #
        #   Format a warning record and append it to the buffered warning list.
        #
        # Arguments:
        #
        #   record : Logging record to format and buffer.
        #
        # Returns:
        #
        #   None.
        #---------------------------------------------------------------------------------------------------------------

        def emit ( self, record ) -> None:

            # Format a warning record and append it to the buffered warning list.

            warning_lines.append ( self.format ( record ) )

    handler = BufferedWarnings ( level = WARNING )
    handler.setFormatter ( Formatter ( "%(levelname)s: %(message)s" ) )
    logger.addHandler ( handler )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: show_warnings
    #
    # Description:
    #
    #   Print and clear buffered warnings, reporting whether any messages were shown.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   True when buffered warnings were printed; otherwise False.
    #-------------------------------------------------------------------------------------------------------------------

    def show_warnings () -> bool:

        # Print and clear buffered warnings, reporting whether any messages were shown.

        if warning_lines:
            for line in warning_lines:
                print ( line, file = runtime.stderr )
            runtime.stderr.flush ()
            warning_lines.clear ()

            # Return data to caller.

            return True

        # Return data to caller.

        return False

    try:

        # Display effective settings, then run the complete classification pipeline.

        _print_banner ( options, alpha, runtime.stdout )
        run (
            options.source, options.dest,
            threshold      = options.threshold,
            alpha          = alpha,
            recursive      = options.recursive,
            manifest       = options.manifest,
            model          = options.model,
            output         = runtime.stdout,
            before_summary = show_warnings,
        )
    except ( UsageError, ClusteringError ) as error:
        show_warnings ()
        print ( f"classifyimages: {error}", file = runtime.stderr )

        # Return data to caller.

        return 2
    except ModelError as error:
        show_warnings ()
        print ( f"classifyimages: {error}", file = runtime.stderr )

        # Return data to caller.

        return 3
    except ( WriteError, OSError ) as error:
        show_warnings ()
        print ( f"classifyimages: {error}", file = runtime.stderr )

        # Return data to caller.

        return 4
    finally:
        logger.removeHandler ( handler )
        handler.close ()

    # Return data to caller.

    return 0


if __name__ == "__main__":
    raise SystemExit ( main () )
