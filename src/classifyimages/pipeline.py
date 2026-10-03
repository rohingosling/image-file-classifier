#-----------------------------------------------------------------------------------------------------------------------
# Module:  pipeline.py
# Project: classifyimages
# Version: 1.0.0
# Date:    2023-03-01
# Author:  Rohin Gosling
#
# Description:
#
#   Connect image ingestion, CPU features, deterministic grouping, and guarded output.
#-----------------------------------------------------------------------------------------------------------------------

from collections.abc import Callable
from pathlib         import Path
from typing          import TextIO

from numpy import asarray, stack

from classifyimages.cluster  import assign_names, duplicates, group
from classifyimages.features import extract_batch, fuse, load_session, prepare
from classifyimages.ingest   import iter_rgb, list_images, load_rgb
from classifyimages.limits   import INFERENCE_BATCH_SIZE, MAX_COLLECTION_IMAGES
from classifyimages.progress import ProgressDisplay
from classifyimages.writeout import UsageError, preflight, validate_paths, write_outputs


#-----------------------------------------------------------------------------------------------------------------------
# Function: _step_progress
#
# Description:
#
#   Map one internal step's counted work into an equal share of the displayed stage.
#
# Arguments:
#
#   progress        : Active progress display receiving aggregate stage updates.
#   completed_steps : Number of equally weighted internal steps already complete.
#   total_steps     : Number of equally weighted steps in the displayed stage.
#
# Returns:
#
#   Callback translating internal work counts into displayed stage progress.
#-----------------------------------------------------------------------------------------------------------------------

def _step_progress (
    progress: ProgressDisplay, completed_steps: int, total_steps: int,
) -> Callable [ [ int, int ], None ]:

    # Map one internal step's counted work into an equal share of the displayed stage.

    #-------------------------------------------------------------------------------------------------------------------
    # Function: record
    #
    # Description:
    #
    #   Map completed work from the current internal step into the displayed stage.
    #
    # Arguments:
    #
    #   completed : Number of completed work units in the current step.
    #   total     : Total work units in the current step.
    #
    # Returns:
    #
    #   None.
    #-------------------------------------------------------------------------------------------------------------------

    def record ( completed: int, total: int ) -> None:

        # Map completed work from the current internal step into the displayed stage.

        if total > 0:
            progress.update ( completed_steps * total + completed, total_steps * total )

    # Return data to caller.

    return record


#-----------------------------------------------------------------------------------------------------------------------
# Function: run
#
# Description:
#
#   Compute a full plan before creating outputs, retaining only small feature records between images.
#
# Arguments:
#
#   source         : Input directory whose files must remain unchanged.
#   destination    : Directory for exclusively created classified PNG outputs.
#   threshold      : Nonnegative average-linkage cosine distance cut.
#   alpha          : Content contribution in [0, 1]; color contributes 1 - alpha.
#   recursive      : Whether discovery descends into source subdirectories.
#   manifest       : Optional CSV path; defaults to manifest.csv in the destination.
#   model          : Optional local ONNX model path override.
#   output         : Text stream receiving settings, progress, or summary output.
#   before_summary : Optional callback that prints pending warnings and reports whether it printed any.
#
# Returns:
#
#   None.
#-----------------------------------------------------------------------------------------------------------------------

def run (
    source: Path,
    destination: Path,
    *,
    threshold: float      = 0.35,
    alpha: float          = 0.80,
    recursive: bool       = False,
    manifest: Path | None = None,
    model: Path | None    = None,
    output: TextIO,
    before_summary: Callable [ [], bool ] | None = None,
) -> None:

    # Compute a full plan before creating outputs, retaining only small feature records between images.

    paths          = validate_paths ( source, destination, manifest )
    skipped_files  = []
    readable_paths = []
    relative_paths = []
    image_features = []
    progress       = ProgressDisplay ( output )

    try:

        # Discover candidates and enforce the collection bound before decoding or loading the model.

        progress.begin ( "Decoding and Normalizing Images:" )
        try:
            image_paths = list_images ( paths.source, recursive, skipped_files = skipped_files )
        except OSError as error:
            raise UsageError ( f"Cannot read source: {error}" ) from error
        if len ( image_paths ) > MAX_COLLECTION_IMAGES:
            raise UsageError (
                f"Found {len(image_paths):,} supported-extension files; the tested maximum is "
                f"{MAX_COLLECTION_IMAGES:,} per run (including unreadable files). Split the source into smaller folders."
            )
        progress.update ( 1, 2 )
        attempted       = 0
        record_decoding = _step_progress ( progress, 1, 2 )

        #---------------------------------------------------------------------------------------------------------------
        # Function: record_attempt
        #
        # Description:
        #
        #   Count each decoding attempt and update the decoding stage.
        #
        # Arguments:
        #
        #   None.
        #
        # Returns:
        #
        #   None.
        #---------------------------------------------------------------------------------------------------------------

        def record_attempt () -> None:

            # Count each decoding attempt and update the decoding stage.

            nonlocal attempted
            attempted += 1
            record_decoding ( attempted, len ( image_paths ) )

        # Probe every candidate and retain only readable paths, closing each full-resolution image immediately.

        try:
            for source_path, image in iter_rgb (
                paths.source, recursive, skipped_files = skipped_files,
                image_paths                            = image_paths, on_attempt = record_attempt,
            ):
                image.close ()
                readable_paths.append ( source_path )
        except OSError as error:
            raise UsageError ( f"Cannot read source: {error}" ) from error
        progress.complete ()

        if not readable_paths:
            raise UsageError ( "No readable images found in the source folder" )

        # Reopen readable images and infer bounded batches while retaining only detached feature arrays.

        progress.begin ( "Extracting Features:" )
        session         = load_session ( model )
        prepared_images = []
        try:
            for image_index, source_path in enumerate ( readable_paths, start = 1 ):
                image = load_rgb ( source_path )
                try:
                    prepared_images.append ( prepare ( image ) )
                finally:
                    image.close ()
                relative_paths.append ( source_path.relative_to ( paths.source ).as_posix () )
                if len ( prepared_images ) == INFERENCE_BATCH_SIZE or image_index == len ( readable_paths ):
                    image_features.extend ( extract_batch ( prepared_images, session, alpha ) )
                    prepared_images.clear ()
                    progress.update ( image_index, len ( readable_paths ) )
        except OSError as error:
            raise UsageError ( f"Cannot read source: {error}" ) from error
        progress.complete ()

        # Allocate equal progress shares to fusion, duplicate grouping, clustering, naming, preflight, and output.

        classification_steps = 6
        progress.begin ( "Classifying Images:" )
        vectors = []
        for image_index, item in enumerate ( image_features, start = 1 ):
            vectors.append ( fuse ( item.embedding, item.histogram, alpha ) )
            progress.update ( image_index, classification_steps * len ( image_features ) )
        vectors = stack ( vectors )
        progress.update ( 1, classification_steps )

        # Confirm complete-link duplicate groups before reducing them to semantic representatives.

        duplicate_groups = duplicates (
            relative_paths,
            [ item.dhash for item in image_features ],
            asarray ( [ item.aspect_ratio for item in image_features ] ),
            stack ( [ item.thumbnail for item in image_features ] ),
            progress = _step_progress ( progress, 1, classification_steps ),
        )
        progress.update ( 2, classification_steps )

        # Cluster duplicate representatives and expand their original image memberships.

        clusters = group ( vectors, relative_paths, duplicate_groups, threshold )
        progress.update ( 3, classification_steps )

        # Assign every deterministic output name before checking or creating destination files.

        plan = assign_names (
            vectors, relative_paths, clusters,
            progress = _step_progress ( progress, 3, classification_steps ),
        )
        progress.update ( 4, classification_steps )

        # Check the complete output plan for collisions before the writer creates any directories.

        preflight (
            paths, plan,
            progress = _step_progress ( progress, 4, classification_steps ),
        )
        progress.update ( 5, classification_steps )

        # Re-decode full-resolution sources and create PNGs with durable manifest updates.

        write_outputs (
            paths, plan, skipped_files,
            progress = _step_progress ( progress, 5, classification_steps ),
        )
        progress.complete ()

        # Show buffered warnings after progress completes, then summarize the finished collection.

        print ( file = output )
        output.flush ()
        if before_summary is not None and before_summary ():
            print ( file = output )
        print ( f"- Read {len(plan)} images.", file = output )
        print ( f"- Clustered {len(plan)} into {len(clusters)} groups.", file = output )
        print ( f"- Skipped {len(skipped_files)}.", file = output )
        print ( file = output )
    except BaseException:
        progress.abort ()
        raise
