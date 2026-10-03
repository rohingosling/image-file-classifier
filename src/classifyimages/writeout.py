#-----------------------------------------------------------------------------------------------------------------------
# Module:  writeout.py
# Project: classifyimages
# Version: 1.0.0
# Date:    2023-03-01
# Author:  Rohin Gosling
#
# Description:
#
#   Guard output paths, save full-resolution PNGs exclusively, and retain a manifest of completed writes.
#-----------------------------------------------------------------------------------------------------------------------

from collections.abc import Callable, Sequence
from csv             import writer
from dataclasses     import dataclass
from os              import fsync
from pathlib         import Path
from typing          import TextIO

from PIL.Image import DecompressionBombError, DecompressionBombWarning

from classifyimages.cluster import NamedImage, format_name
from classifyimages.ingest  import SkippedFile, load_rgb

# CSV manifest fields in output order.

MANIFEST_HEADER = ( "cluster", "index", "output_name", "source_path", "distance_to_centroid", "skipped_reason" )


#-----------------------------------------------------------------------------------------------------------------------
# Class: UsageError
#
# Description:
#
#   Invalid source/output paths or a collision, mapped to exit code 2.
#-----------------------------------------------------------------------------------------------------------------------

class UsageError ( ValueError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: WriteError
#
# Description:
#
#   An output could not be created or completed, mapped to exit code 4.
#-----------------------------------------------------------------------------------------------------------------------

class WriteError ( OSError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: OutputPaths
#
# Description:
#
#   Resolved locations shared by planning and output; constructing them creates nothing.
#
# Attributes:
#
#   source      : Resolved input directory.
#   destination : Resolved directory for classified PNG outputs.
#   manifest    : Resolved CSV manifest path.
#-----------------------------------------------------------------------------------------------------------------------

@dataclass ( frozen = True )
class OutputPaths:

    source:      Path
    destination: Path
    manifest:    Path


#-----------------------------------------------------------------------------------------------------------------------
# Function: _occupied
#
# Description:
#
#   Include dangling links and directories when refusing an existing output name.
#
# Arguments:
#
#   path : Filesystem path to inspect.
#
# Returns:
#
#   True when any filesystem entry occupies the path, including a dangling link.
#-----------------------------------------------------------------------------------------------------------------------

def _occupied ( path: Path ) -> bool:

    # Include dangling links and directories when refusing an existing output name.

    try:
        path.lstat ()
    except FileNotFoundError:

        # Return data to caller.

        return False

    # Return data to caller.

    return True


#-----------------------------------------------------------------------------------------------------------------------
# Function: _directory_ancestors
#
# Description:
#
#   Reject a file used as a directory before creating any part of the output tree.
#
# Arguments:
#
#   path : Filesystem path to inspect.
#
# Returns:
#
#   None.
#-----------------------------------------------------------------------------------------------------------------------

def _directory_ancestors ( path: Path ) -> None:

    # Reject a file used as a directory before creating any part of the output tree.

    for directory in ( path, *path.parents ):
        if _occupied ( directory ) and not directory.is_dir ():
            raise UsageError ( f"Not a directory: {directory}" )


#-----------------------------------------------------------------------------------------------------------------------
# Function: validate_paths
#
# Description:
#
#   Protect source files, including through Windows junctions and case-insensitive path aliases.
#
# Arguments:
#
#   source      : Input directory whose files must remain unchanged.
#   destination : Directory for exclusively created classified PNG outputs.
#   manifest    : Optional CSV path; defaults to manifest.csv in the destination.
#
# Returns:
#
#   Resolved source, destination, and manifest locations.
#-----------------------------------------------------------------------------------------------------------------------

def validate_paths ( source: Path, destination: Path, manifest: Path | None = None ) -> OutputPaths:

    # Protect source files, including through Windows junctions and case-insensitive path aliases.

    try:
        source = source.expanduser ().resolve ( strict = True )
        if not source.is_dir ():
            raise UsageError ( f"Source is not a directory: {source}" )
    except ( OSError, RuntimeError, ValueError ) as error:
        raise UsageError ( f"Cannot read source: {error}" ) from error

    # Reject occupied manifest paths and resolve aliases before checking source containment.

    try:
        destination = destination.expanduser ()
        _directory_ancestors ( destination )
        destination = destination.resolve ()
        manifest    = manifest.expanduser () if manifest is not None else destination / "manifest.csv"
        if _occupied ( manifest ):
            raise UsageError ( f"Manifest already exists: {manifest}" )
        manifest = manifest.resolve ()
        _directory_ancestors ( manifest.parent )
        if destination == manifest or destination.is_relative_to ( manifest ):
            raise UsageError ( "Manifest must be a file, not the destination or one of its parents" )
        if destination.is_relative_to ( source ):
            raise UsageError ( "Destination must be outside the source folder" )
        if manifest.is_relative_to ( source ):
            raise UsageError ( "Manifest must be outside the source folder" )
    except ( RuntimeError, ValueError ) as error:
        raise UsageError ( str ( error ) ) from error
    except OSError as error:
        raise WriteError ( f"Cannot inspect output paths: {error}" ) from error

    # Return data to caller.

    return OutputPaths ( source, destination, manifest )


#-----------------------------------------------------------------------------------------------------------------------
# Function: preflight
#
# Description:
#
#   Check the complete set before mkdir or any writes; repeat exclusive checks when opening files.
#
# Arguments:
#
#   paths    : Resolved source, destination, and manifest paths.
#   plan     : Complete ordered PNG output plan.
#   progress : Optional callback receiving completed and total work counts.
#
# Returns:
#
#   None.
#-----------------------------------------------------------------------------------------------------------------------

def preflight (
    paths: OutputPaths, plan: Sequence [ NamedImage ], *,
    progress: Callable [ [ int, int ], None ] | None = None,
) -> None:

    # Check the complete set before mkdir or any writes; repeat exclusive checks when opening files.

    targets = set ()
    try:
        for image_index, item in enumerate ( plan, start = 1 ):
            if item.output_name != format_name ( item.cluster, item.index ):
                raise UsageError ( "Output plan contains an invalid PNG name" )
            target = paths.destination / item.output_name
            if target in targets:
                raise UsageError ( f"Output plan contains a repeated name: {target.name}" )
            targets.add ( target )
            if paths.manifest == target or paths.manifest.is_relative_to ( target ):
                raise UsageError ( f"Manifest conflicts with PNG output: {target}" )
            if _occupied ( target ):
                raise UsageError ( f"Output already exists: {target}" )
            if progress is not None:
                progress ( image_index, len ( plan ) )
        if _occupied ( paths.manifest ):
            raise UsageError ( f"Manifest already exists: {paths.manifest}" )
    except OSError as error:
        raise WriteError ( f"Cannot inspect output names: {error}" ) from error


#-----------------------------------------------------------------------------------------------------------------------
# Function: image_row
#
# Description:
#
#   Store normalized source-relative paths, including CSV-sensitive characters losslessly.
#
# Arguments:
#
#   item : Planned output image with its source path and ordering metadata.
#
# Returns:
#
#   CSV-compatible tuple containing the output mapping and centroid distance.
#-----------------------------------------------------------------------------------------------------------------------

def image_row ( item: NamedImage ) -> tuple:

    # Store normalized source-relative paths, including CSV-sensitive characters losslessly.

    # Return data to caller.

    return ( item.cluster, item.index, item.output_name, item.relative_path, format ( item.distance_to_centroid, ".17g" ), "" )


#-----------------------------------------------------------------------------------------------------------------------
# Function: write_manifest
#
# Description:
#
#   Write the CSV header, skipped-file records, and completed image rows to an open manifest.
#
# Arguments:
#
#   stream        : Open text stream for the CSV manifest.
#   plan          : Complete ordered PNG output plan.
#   skipped_files : Collection of skipped-file records.
#
# Returns:
#
#   None.
#-----------------------------------------------------------------------------------------------------------------------

def write_manifest ( stream: TextIO, plan: Sequence [ NamedImage ], skipped_files: Sequence [ SkippedFile ] ) -> None:

    # Write the CSV header, skipped-file records, and completed image rows to an open manifest.

    rows = writer ( stream, lineterminator = "\n" )
    rows.writerow ( MANIFEST_HEADER )
    for skipped in sorted ( skipped_files, key = lambda item: item.relative_path ):
        rows.writerow ( ( "", "", "", skipped.relative_path, "", skipped.reason ) )
    rows.writerows ( image_row ( item ) for item in plan )


#-----------------------------------------------------------------------------------------------------------------------
# Function: _sync
#
# Description:
#
#   Finish each CSV update before moving to the next PNG.
#
# Arguments:
#
#   stream : Open text stream for the CSV manifest.
#
# Returns:
#
#   None.
#-----------------------------------------------------------------------------------------------------------------------

def _sync ( stream: TextIO ) -> None:

    # Finish each CSV update before moving to the next PNG.

    stream.flush ()
    fsync ( stream.fileno () )


#-----------------------------------------------------------------------------------------------------------------------
# Function: write_png
#
# Description:
#
#   Re-decode one full-resolution image; exclusive creation never replaces a preexisting file.
#
# Arguments:
#
#   source_path : Path of the original image file.
#   target      : PNG output path to create exclusively.
#
# Returns:
#
#   None.
#-----------------------------------------------------------------------------------------------------------------------

def write_png ( source_path: Path, target: Path ) -> None:

    # Re-decode one full-resolution image; exclusive creation never replaces a preexisting file.

    with load_rgb ( source_path ) as image:
        with target.open ( "xb" ) as output:
            try:
                image.save ( output, format = "PNG" )
                output.flush ()
                fsync ( output.fileno () )
            except BaseException:
                output.close ()
                target.unlink ( missing_ok = True )
                raise


#-----------------------------------------------------------------------------------------------------------------------
# Function: write_outputs
#
# Description:
#
#   Leave completed images and flushed CSV rows on failure, removing only an incomplete new image.
#
#   Failure of the manifest device itself may prevent recovery of its last row; report that as a write failure. Source
#   folders must remain stable throughout a run, including between feature extraction and re-decoding.
#
# Arguments:
#
#   paths         : Resolved source, destination, and manifest paths.
#   plan          : Complete ordered PNG output plan.
#   skipped_files : Collection of skipped-file records.
#   progress      : Optional callback receiving completed and total work counts.
#
# Returns:
#
#   None.
#-----------------------------------------------------------------------------------------------------------------------

def write_outputs (
    paths: OutputPaths, plan: Sequence [ NamedImage ], skipped_files: Sequence [ SkippedFile ],
    *, progress: Callable [ [ int, int ], None ] | None = None,
) -> None:

    # Leave completed images and flushed CSV rows on failure, removing only an incomplete new image.

    preflight ( paths, plan )
    try:
        paths.destination.mkdir ( parents = True, exist_ok = True )
        paths.manifest.parent.mkdir ( parents = True, exist_ok = True )
        with paths.manifest.open ( "x", encoding = "utf-8", newline = "" ) as stream:
            write_manifest ( stream, (), skipped_files )
            _sync ( stream )

            # Append each successful PNG only after its pixels have been flushed to disk.

            rows = writer ( stream, lineterminator = "\n" )
            for image_index, item in enumerate ( plan, start = 1 ):
                target     = paths.destination / item.output_name
                checkpoint = stream.tell ()
                write_png ( paths.source / item.relative_path, target )
                try:
                    rows.writerow ( image_row ( item ) )
                    _sync ( stream )
                except BaseException:

                    # Remove the unmatched PNG and attempt to restore the last complete manifest checkpoint.

                    target.unlink ( missing_ok = True )
                    try:
                        stream.seek ( checkpoint )
                        stream.truncate ()
                        _sync ( stream )
                    except OSError:
                        pass
                    raise
                if progress is not None:
                    progress ( image_index, len ( plan ) )
    except FileExistsError as error:
        raise UsageError ( f"Output appeared after collision checks; nothing was overwritten: {error}" ) from error
    except ( OSError, ValueError, SyntaxError, DecompressionBombError, DecompressionBombWarning ) as error:
        raise WriteError ( f"Write failed; inspect partial output in {paths.destination} and manifest {paths.manifest}: {error}" ) from error
