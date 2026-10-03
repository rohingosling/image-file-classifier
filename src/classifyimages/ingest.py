#-----------------------------------------------------------------------------------------------------------------------
# Module:  ingest.py
# Project: classifyimages
# Version: 1.0.0
# Date:    2023-03-01
# Author:  Rohin Gosling
#
# Description:
#
#   Discover images deterministically, decode full-resolution RGB pixels, and encode PNG bytes without source writes.
#-----------------------------------------------------------------------------------------------------------------------

from collections.abc import Callable, Iterator, Sequence
from dataclasses     import dataclass
from io              import BytesIO
from logging         import getLogger
from os              import walk
from pathlib         import Path
from warnings        import catch_warnings, simplefilter

from PIL.Image    import Image, DecompressionBombError, DecompressionBombWarning, new, open as open_image
from PIL.ImageOps import exif_transpose

# Supported image extensions and module logger.

SUPPORTED_EXTENSIONS = frozenset ( { ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff" } )
LOGGER               = getLogger ( __name__ )


#-----------------------------------------------------------------------------------------------------------------------
# Class: SkippedFile
#
# Description:
#
#   Retain both filesystem and normalized relative paths for logging and the manifest.
#
# Attributes:
#
#   source_path   : Filesystem path of the skipped source file.
#   relative_path : Normalized forward-slash source-relative path.
#   reason        : Explanation for skipping the file.
#-----------------------------------------------------------------------------------------------------------------------

@dataclass ( frozen = True )
class SkippedFile:

    source_path:   Path
    relative_path: str
    reason:        str


#-----------------------------------------------------------------------------------------------------------------------
# Function: _record_skip
#
# Description:
#
#   Report one skipped input and optionally retain its manifest data.
#
# Arguments:
#
#   source_path   : Path of the original image file.
#   source        : Input directory whose files must remain unchanged.
#   reason        : Explanation retained for logging or the manifest.
#   skipped_files : Optional collection in which to retain the skipped-file record.
#
# Returns:
#
#   None.
#-----------------------------------------------------------------------------------------------------------------------

def _record_skip ( source_path: Path, source: Path, reason: str, skipped_files: list [ SkippedFile ] | None ) -> None:

    # Report one skipped input and optionally retain its manifest data.

    relative_path = source_path.relative_to ( source ).as_posix ()
    if skipped_files is not None:
        skipped_files.append ( SkippedFile ( source_path, relative_path, reason ) )
    LOGGER.warning ( "Skipping %s: %s", relative_path, reason )


#-----------------------------------------------------------------------------------------------------------------------
# Function: _raise_discovery_error
#
# Description:
#
#   Do not silently return an incomplete collection when a directory cannot be scanned.
#
# Arguments:
#
#   error : Filesystem discovery error to propagate.
#
# Returns:
#
#   None.
#-----------------------------------------------------------------------------------------------------------------------

def _raise_discovery_error ( error: OSError ) -> None:

    # Do not silently return an incomplete collection when a directory cannot be scanned.

    raise error


#-----------------------------------------------------------------------------------------------------------------------
# Function: list_images
#
# Description:
#
#   Return supported files sorted by forward-slash source-relative path, recording unsupported files.
#
#   Returned paths are absolute. Recursive discovery does not follow directory symlinks or Windows junctions. Missing or
#   unreadable source directories raise OSError; decoding is deferred to load_rgb or iter_rgb.
#
# Arguments:
#
#   source        : Input directory whose files must remain unchanged.
#   recursive     : Whether discovery descends into source subdirectories.
#   skipped_files : Optional collection in which to retain unsupported-file records.
#
# Returns:
#
#   Absolute supported-image paths sorted by normalized source-relative path.
#-----------------------------------------------------------------------------------------------------------------------

def list_images ( source: Path, recursive: bool = False, *, skipped_files: list [ SkippedFile ] | None = None ) -> list [ Path ]:

    # Return supported files sorted by forward-slash source-relative path, recording unsupported files.

    source = Path ( source ).resolve ( strict = True )
    if not source.is_dir ():
        raise NotADirectoryError ( f"Source is not a directory: {source}" )

    # Gather paths before filtering so discovery and skip reporting have the same stable ordering.

    if recursive:
        discovered_paths = []
        for directory, directory_names, file_names in walk ( source, onerror = _raise_discovery_error, followlinks = False ):
            directory_path = Path ( directory )
            directory_names [ : ] = [
                name for name in directory_names
                if not ( directory_path / name ).is_symlink () and not ( directory_path / name ).is_junction ()
            ]
            discovered_paths.extend ( directory_path / name for name in file_names )
    else:
        discovered_paths = [ path for path in source.iterdir () if not path.is_dir () ]

    image_paths = []
    for source_path in sorted ( discovered_paths, key = lambda path: path.relative_to ( source ).as_posix () ):
        if source_path.suffix.lower () in SUPPORTED_EXTENSIONS:
            image_paths.append ( source_path )
        else:
            extension = source_path.suffix or "(none)"
            _record_skip ( source_path, source, f"unsupported extension: {extension}", skipped_files )

    # Return the deterministic candidate list without decoding pixels.

    return image_paths


#-----------------------------------------------------------------------------------------------------------------------
# Function: load_rgb
#
# Description:
#
#   Read the first frame, apply EXIF orientation, and detach full-resolution RGB pixels from the source.
#
#   Transparency is composited onto white, including palette and color-key transparency. Metadata is discarded so
#   transparency or orientation cannot be reapplied to the converted PNG. The caller owns and must close the result.
#   Pillow's strict truncated-file policy is retained. Decode failures propagate to the per-file boundary in iter_rgb.
#
# Arguments:
#
#   source_path : Path of the original image file.
#
# Returns:
#
#   Independently owned full-resolution RGB image that the caller must close.
#-----------------------------------------------------------------------------------------------------------------------

def load_rgb ( source_path: Path ) -> Image:

    # Read the first frame, apply EXIF orientation, and detach full-resolution RGB pixels from the source.

    with catch_warnings ():
        simplefilter ( "error", DecompressionBombWarning )

        # Verify structure before reopening for decoding; PNG verification also checks chunk integrity.

        with open_image ( source_path ) as probe:
            probe.verify ()
        with open_image ( source_path ) as decoded:
            decoded.load ()
            exif_transpose ( decoded, in_place = True )
            if "A" in decoded.getbands () or "transparency" in decoded.info:
                with decoded.convert ( "RGBA" ) as transparent:
                    converted = new ( "RGB", decoded.size, "white" )
                    with transparent.getchannel ( "A" ) as alpha_channel:
                        converted.paste ( transparent, mask = alpha_channel )
            else:
                converted = decoded.convert ( "RGB" )
            converted.info.clear ()

    # Return independently owned pixels after closing every source handle.

    return converted


#-----------------------------------------------------------------------------------------------------------------------
# Function: iter_rgb
#
# Description:
#
#   Yield one readable source path and RGB image at a time, logging and retaining per-file decode failures.
#
#   The consumer must close each yielded image before requesting the next to keep full-resolution memory bounded. An
#   empty iterator means no readable images; the CLI maps that result to exit code 2.
#
# Arguments:
#
#   source        : Input directory whose files must remain unchanged.
#   recursive     : Whether discovery descends into source subdirectories.
#   skipped_files : Optional collection in which to retain unsupported or unreadable-file records.
#   image_paths   : Optional precomputed sequence of supported candidate paths.
#   on_attempt    : Optional callback invoked after every decoding attempt.
#
# Returns:
#
#   Iterator yielding readable source paths and independently owned RGB images.
#-----------------------------------------------------------------------------------------------------------------------

def iter_rgb (
    source: Path,
    recursive: bool = False,
    *,
    skipped_files: list [ SkippedFile ] | None = None,
    image_paths: Sequence [ Path ] | None      = None,
    on_attempt: Callable [ [], None ] | None   = None,
) -> Iterator [ tuple [ Path, Image ] ]:

    # Yield one readable source path and RGB image at a time, logging and retaining per-file decode failures.

    source = Path ( source ).resolve ( strict = True )
    for source_path in image_paths if image_paths is not None else list_images ( source, recursive, skipped_files = skipped_files ):
        try:
            converted = load_rgb ( source_path )
        except ( OSError, ValueError, SyntaxError, DecompressionBombError, DecompressionBombWarning ) as error:
            _record_skip ( source_path, source, f"unreadable image: {error}", skipped_files )
            if on_attempt is not None:
                on_attempt ()
            continue
        if on_attempt is not None:
            on_attempt ()
        yield source_path, converted


#-----------------------------------------------------------------------------------------------------------------------
# Function: png_bytes
#
# Description:
#
#   Encode an already oriented, white-composited RGB image without resizing or touching the filesystem.
#
# Arguments:
#
#   image : Nonempty oriented RGB image owned by the caller.
#
# Returns:
#
#   Detached full-resolution PNG-encoded bytes.
#-----------------------------------------------------------------------------------------------------------------------

def png_bytes ( image: Image ) -> bytes:

    # Encode an already oriented, white-composited RGB image without resizing or touching the filesystem.

    if image.mode != "RGB":
        raise ValueError ( "PNG encoding requires an RGB image from load_rgb" )
    with BytesIO () as output:
        image.save ( output, format = "PNG" )

        # Return a detached byte string before closing the in-memory stream.

        return output.getvalue ()
