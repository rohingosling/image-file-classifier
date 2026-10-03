#-----------------------------------------------------------------------------------------------------------------------
# Module:  paths.py
# Project: classifyimages
# Version: 0.1.0
# Date:    2023-03-01
# Author:  Rohin Gosling
#
# Description:
#
#   Resolve a model override, PyInstaller payload, or source-tree model without a download fallback.
#-----------------------------------------------------------------------------------------------------------------------

from importlib import import_module
from pathlib   import Path


#-----------------------------------------------------------------------------------------------------------------------
# Function: resolve_model_path
#
# Description:
#
#   Resolve an explicit local override, or locate bundled model data independently of the working directory.
#   Bundled model lookup is also independent of the executable's containing folder.
#
# Arguments:
#
#   override : Optional local model path supplied by the caller.
#
# Returns:
#
#   Resolved local model path without checking model validity or downloading data.
#-----------------------------------------------------------------------------------------------------------------------

def resolve_model_path ( override: Path | None = None ) -> Path:

    # Resolve an explicit local override, or locate bundled model data independently of the working directory.

    if override is not None:

        # Return data to caller.

        return Path ( override ).expanduser ().resolve ()
    runtime = import_module ( "sys" )
    if getattr ( runtime, "frozen", False ):
        extraction_directory = getattr ( runtime, "_MEIPASS", None )
        if extraction_directory is None:
            raise RuntimeError ( "Frozen application has no model extraction directory" )
        resource_root = Path ( extraction_directory )
    else:
        resource_root = Path ( __file__ ).resolve ().parents [ 2 ]

    # Return only the selected local path; a missing model must fail at the load boundary.

    return resource_root / "models" / "mobilenet_v3_small.onnx"
