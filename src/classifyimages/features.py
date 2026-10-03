#-----------------------------------------------------------------------------------------------------------------------
# Module:  features.py
# Project: classifyimages
# Version: 0.1.0
# Date:    2023-03-01
# Author:  Rohin Gosling
#
# Description:
#
#   Extract CPU ONNX embeddings, bounded HSV histograms, duplicate signals, and weighted feature vectors.
#-----------------------------------------------------------------------------------------------------------------------

from collections.abc import Sequence
from dataclasses     import dataclass
from math            import isfinite as finite_scalar, sqrt
from os              import cpu_count
from pathlib         import Path

from numpy        import asarray, ascontiguousarray, bincount, concatenate, float32, float64, isfinite, ndarray, packbits, stack
from numpy.linalg import norm
from onnxruntime  import InferenceSession, SessionOptions, disable_telemetry_events
from PIL.Image    import Image, Resampling, new

from classifyimages.paths  import resolve_model_path
from classifyimages.limits import INFERENCE_BATCH_SIZE

# Embedding dimensions, HSV marginal bins, and ImageNet normalization constants.

EMBEDDING_SIZE              = 576
HISTOGRAM_BINS              = ( 16, 8, 8 )
IMAGENET_MEAN               = asarray ( ( 0.485, 0.456, 0.406 ), dtype = float32 ) [ :, None, None ]
IMAGENET_STANDARD_DEVIATION = asarray ( ( 0.229, 0.224, 0.225 ), dtype = float32 ) [ :, None, None ]


#-----------------------------------------------------------------------------------------------------------------------
# Class: ModelError
#
# Description:
#
#   Local model load, signature, or inference failure; the CLI maps this boundary to exit code 3.
#-----------------------------------------------------------------------------------------------------------------------

class ModelError ( RuntimeError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: ImageFeatures
#
# Description:
#
#   Detached feature arrays and duplicate evidence, retaining no full-resolution Pillow image.
#
# Attributes:
#
#   embedding    : Unit pooled content embedding.
#   histogram    : Unit concatenation of HSV marginal histograms.
#   vector       : Unit weighted content-and-color feature vector.
#   dhash        : Unsigned 64-bit horizontal grayscale difference hash.
#   aspect_ratio : Oriented image width divided by its height.
#   thumbnail    : White-padded 64 by 64 float32 RGB thumbnail in [0, 1].
#-----------------------------------------------------------------------------------------------------------------------

@dataclass ( frozen = True )
class ImageFeatures:

    embedding:    ndarray
    histogram:    ndarray
    vector:       ndarray
    dhash:        int
    aspect_ratio: float
    thumbnail:    ndarray


#-----------------------------------------------------------------------------------------------------------------------
# Class: PreparedImage
#
# Description:
#
#   Small detached model input and duplicate evidence; the full-resolution image can be closed immediately.
#
# Attributes:
#
#   inputs       : Detached float32 model input with shape [1, 3, 224, 224].
#   histogram    : Unit concatenation of HSV marginal histograms.
#   dhash        : Unsigned 64-bit horizontal grayscale difference hash.
#   aspect_ratio : Oriented image width divided by its height.
#   thumbnail    : White-padded 64 by 64 float32 RGB thumbnail in [0, 1].
#-----------------------------------------------------------------------------------------------------------------------

@dataclass ( frozen = True )
class PreparedImage:

    inputs:       ndarray
    histogram:    ndarray
    dhash:        int
    aspect_ratio: float
    thumbnail:    ndarray


#-----------------------------------------------------------------------------------------------------------------------
# Function: _require_rgb
#
# Description:
#
#   Require the orientation and alpha conversion contract supplied by ingest.load_rgb.
#
# Arguments:
#
#   image : Nonempty oriented RGB image owned by the caller.
#
# Returns:
#
#   None.
#-----------------------------------------------------------------------------------------------------------------------

def _require_rgb ( image: Image ) -> None:

    # Require the orientation and alpha conversion contract supplied by ingest.load_rgb.

    if image.mode != "RGB" or min ( image.size ) < 1:
        raise ValueError ( "Feature extraction requires a nonempty RGB image from load_rgb" )


#-----------------------------------------------------------------------------------------------------------------------
# Function: _normalize
#
# Description:
#
#   Return a finite unit float32 vector, rejecting degenerate data instead of inventing a direction.
#
# Arguments:
#
#   vector : One finite, nonzero feature vector.
#
# Returns:
#
#   Contiguous finite unit float32 vector.
#-----------------------------------------------------------------------------------------------------------------------

def _normalize ( vector: ndarray ) -> ndarray:

    # Return a finite unit float32 vector, rejecting degenerate data instead of inventing a direction.

    values = asarray ( vector, dtype = float64 )
    if values.ndim != 1 or values.size == 0 or not isfinite ( values ).all ():
        raise ValueError ( "Features must be a nonempty, finite one-dimensional vector" )
    magnitude = norm ( values )
    if not finite_scalar ( magnitude ) or magnitude == 0:
        raise ValueError ( "Features must have a finite nonzero L2 norm" )

    # Normalize in float64 to avoid overflow or underflow for finite float32 model outputs.

    return ascontiguousarray ( values / magnitude, dtype = float32 )


#-----------------------------------------------------------------------------------------------------------------------
# Function: preprocess
#
# Description:
#
#   Match IMAGENET1K_V1's Pillow bilinear resize and rounded center crop; return float32 [1, 3, 224, 224].
#
# Arguments:
#
#   image : Nonempty oriented RGB image owned by the caller.
#
# Returns:
#
#   Detached contiguous float32 batch with shape [1, 3, 224, 224].
#-----------------------------------------------------------------------------------------------------------------------

def preprocess ( image: Image ) -> ndarray:

    # Match IMAGENET1K_V1's Pillow bilinear resize and rounded center crop; return float32 [1, 3, 224, 224].

    _require_rgb ( image )
    width, height = image.size
    if width <= height:
        resized_size = ( 256, int ( 256 * height / width ) )
    else:
        resized_size = ( int ( 256 * width / height ), 256 )
    left = int ( round ( ( resized_size [ 0 ] - 224 ) / 2.0 ) )
    top  = int ( round ( ( resized_size [ 1 ] - 224 ) / 2.0 ) )
    with image.resize ( resized_size, Resampling.BILINEAR ) as resized:
        with resized.crop ( ( left, top, left + 224, top + 224 ) ) as cropped:
            channels = asarray ( cropped, dtype = float32 ).transpose ( 2, 0, 1 ) / float32 ( 255.0 )

    # Standardize channel values using the pretrained ImageNet weights' input statistics.

    channels = ( channels - IMAGENET_MEAN ) / IMAGENET_STANDARD_DEVIATION

    # Return a detached, contiguous batch without changing the source pixels or dimensions.

    return ascontiguousarray ( channels [ None, : ], dtype = float32 )


#-----------------------------------------------------------------------------------------------------------------------
# Function: _preview
#
# Description:
#
#   Resize before color conversion or array allocation, bounding all downstream preview work.
#
# Arguments:
#
#   image        : Nonempty oriented RGB image owned by the caller.
#   longest_edge : Maximum preview width or height in pixels.
#   upscale      : Whether previews may enlarge images smaller than the requested edge.
#
# Returns:
#
#   Independently owned RGB preview that the caller must close.
#-----------------------------------------------------------------------------------------------------------------------

def _preview ( image: Image, longest_edge: int, *, upscale: bool = False ) -> Image:

    # Resize before color conversion or array allocation, bounding all downstream preview work.

    scale = longest_edge / max ( image.size )
    if not upscale:
        scale = min ( 1.0, scale )
    size = tuple ( max ( 1, round ( dimension * scale ) ) for dimension in image.size )

    # Return an independently owned Pillow preview that the caller must close.

    return image.resize ( size, Resampling.BILINEAR )


#-----------------------------------------------------------------------------------------------------------------------
# Function: histogram
#
# Description:
#
#   Return 32 unit HSV marginal bins from an aspect-preserving preview whose longest edge is at most 256.
#
# Arguments:
#
#   image : Nonempty oriented RGB image owned by the caller.
#
# Returns:
#
#   Unit float32 vector containing 16 hue, 8 saturation, and 8 value bins.
#-----------------------------------------------------------------------------------------------------------------------

def histogram ( image: Image ) -> ndarray:

    # Return 32 unit HSV marginal bins from an aspect-preserving preview whose longest edge is at most 256.

    _require_rgb ( image )
    with _preview ( image, 256 ) as preview:
        with preview.convert ( "HSV" ) as converted:
            pixels = asarray ( converted ).reshape ( -1, 3 )
            channels = [
                bincount ( pixels [ :, channel ] // ( 256 // bin_count ), minlength = bin_count ) / len ( pixels )
                for channel, bin_count in enumerate ( HISTOGRAM_BINS )
            ]

    # Normalize after concatenating the three independently probability-normalized histograms.

    return _normalize ( concatenate ( channels ) )


#-----------------------------------------------------------------------------------------------------------------------
# Function: dhash
#
# Description:
#
#   Return 64 horizontal grayscale comparisons; left > right sets a bit, with the first pixel at bit 63.
#
# Arguments:
#
#   image : Nonempty oriented RGB image owned by the caller.
#
# Returns:
#
#   Unsigned 64-bit horizontal grayscale difference hash.
#-----------------------------------------------------------------------------------------------------------------------

def dhash ( image: Image ) -> int:

    # Return 64 horizontal grayscale comparisons; left > right sets a bit, with the first pixel at bit 63.

    _require_rgb ( image )
    with image.convert ( "L" ) as grayscale:
        with grayscale.resize ( ( 9, 8 ), Resampling.BILINEAR ) as resized:
            pixels      = asarray ( resized )
            comparisons = pixels [ :, : -1 ] > pixels [ :, 1 : ]

    # Keep all 64 bits in a Python integer for XOR and bit_count during duplicate nomination.

    return int.from_bytes ( packbits ( comparisons, bitorder = "big" ).tobytes (), byteorder = "big" )


#-----------------------------------------------------------------------------------------------------------------------
# Function: verification_thumbnail
#
# Description:
#
#   Return a centered, white-padded 64 by 64 RGB thumbnail as float32 HWC values in [0, 1].
#
# Arguments:
#
#   image : Nonempty oriented RGB image owned by the caller.
#
# Returns:
#
#   Detached float32 RGB thumbnail with shape [64, 64, 3] in [0, 1].
#-----------------------------------------------------------------------------------------------------------------------

def verification_thumbnail ( image: Image ) -> ndarray:

    # Return a centered, white-padded 64 by 64 RGB thumbnail as float32 HWC values in [0, 1].

    _require_rgb ( image )
    with _preview ( image, 64, upscale = True ) as preview:
        with new ( "RGB", ( 64, 64 ), "white" ) as padded:
            padded.paste ( preview, ( ( 64 - preview.width ) // 2, ( 64 - preview.height ) // 2 ) )
            pixels = asarray ( padded, dtype = float32 ) / float32 ( 255.0 )

    # Return detached pixels for the mean-absolute-RGB-error guard.

    return pixels


#-----------------------------------------------------------------------------------------------------------------------
# Function: fuse
#
# Description:
#
#   Give content and color their requested cosine contributions after separately normalizing both blocks.
#
# Arguments:
#
#   embedding       : Content embedding to normalize and weight.
#   color_histogram : Color histogram to normalize and weight.
#   alpha           : Content contribution in [0, 1]; color contributes 1 - alpha.
#
# Returns:
#
#   Unit concatenation of separately normalized and square-root-weighted feature blocks.
#-----------------------------------------------------------------------------------------------------------------------

def fuse ( embedding: ndarray, color_histogram: ndarray, alpha: float = 0.80 ) -> ndarray:

    # Give content and color their requested cosine contributions after separately normalizing both blocks.

    if not finite_scalar ( alpha ) or not 0.0 <= alpha <= 1.0:
        raise ValueError ( "alpha must be a finite number in [0, 1]" )
    content = _normalize ( embedding )
    color   = _normalize ( color_histogram )

    # Square-root scales preserve alpha as a similarity contribution, including exactly zero inactive blocks at both
    # endpoints.

    return _normalize ( concatenate ( ( sqrt ( alpha ) * content, sqrt ( 1.0 - alpha ) * color ) ) )


#-----------------------------------------------------------------------------------------------------------------------
# Function: _validate_signature
#
# Description:
#
#   Accept exactly one dynamic float32 NCHW image input and one dynamic 576-value embedding output.
#
# Arguments:
#
#   session : Reusable local CPU ONNX inference session.
#
# Returns:
#
#   None.
#-----------------------------------------------------------------------------------------------------------------------

def _validate_signature ( session: InferenceSession ) -> None:

    # Accept exactly one dynamic float32 NCHW image input and one dynamic 576-value embedding output.

    model_inputs  = session.get_inputs ()
    model_outputs = session.get_outputs ()
    if len ( model_inputs ) != 1 or len ( model_outputs ) != 1:
        raise ModelError ( "Model must have exactly one input and one output" )
    for tensor, dimensions, label in (
        ( model_inputs [ 0 ], [ 3, 224, 224 ], "input" ),
        ( model_outputs [ 0 ], [ EMBEDDING_SIZE ], "output" ),
    ):
        shape = tensor.shape
        if tensor.type != "tensor(float)" or shape is None or len ( shape ) != len ( dimensions ) + 1:
            raise ModelError ( f"Model {label} must be float32 with shape [batch, {', '.join(map(str, dimensions))}]" )
        if shape [ 1 : ] != dimensions or not ( shape [ 0 ] is None or isinstance ( shape [ 0 ], str ) and bool ( shape [ 0 ] ) ):
            raise ModelError ( f"Model {label} has incompatible shape {shape}; a dynamic batch dimension is required" )
    if session.get_providers () != [ "CPUExecutionProvider" ]:
        raise ModelError ( "Model must use CPUExecutionProvider exclusively" )


#-----------------------------------------------------------------------------------------------------------------------
# Function: load_session
#
# Description:
#
#   Load only local model data and validate the signature before any image inference.
#
# Arguments:
#
#   model : Optional local ONNX model path override.
#
# Returns:
#
#   Validated reusable ONNX session using CPUExecutionProvider exclusively.
#-----------------------------------------------------------------------------------------------------------------------

def load_session ( model: Path | None = None ) -> InferenceSession:

    # Load only local model data and validate the signature before any image inference.

    try:
        selected_path = resolve_model_path ( model )
        if not selected_path.is_file ():
            raise ModelError ( f"Model file does not exist: {selected_path}" )

        # Disable telemetry and bound CPU threading before creating the local inference session.

        disable_telemetry_events ()
        options                      = SessionOptions ()
        options.intra_op_num_threads = min ( 4, cpu_count () or 1 )
        session                      = InferenceSession ( str ( selected_path ), sess_options = options, providers = [ "CPUExecutionProvider" ] )
        _validate_signature ( session )
    except ModelError:
        raise
    except Exception as error:
        raise ModelError ( f"Unable to load local ONNX model: {error}" ) from error

    # Return one reusable CPU session for the entire image collection.

    return session


#-----------------------------------------------------------------------------------------------------------------------
# Function: embed_batch
#
# Description:
#
#   Infer at most eight prepared images and reject malformed output rows before returning any embeddings.
#
# Arguments:
#
#   inputs  : Prepared float32 image batch with shape [batch, 3, 224, 224].
#   session : Reusable local CPU ONNX inference session.
#
# Returns:
#
#   One unit float32 embedding row per prepared image.
#-----------------------------------------------------------------------------------------------------------------------

def embed_batch ( inputs: ndarray, session: InferenceSession ) -> ndarray:

    # Infer at most eight prepared images and reject malformed output rows before returning any embeddings.

    if inputs.ndim != 4 or inputs.shape [ 1 : ] != ( 3, 224, 224 ) or inputs.dtype != float32:
        raise ValueError ( "Model inputs must be float32 [batch, 3, 224, 224]" )
    batch_size = len ( inputs )
    if not 1 <= batch_size <= INFERENCE_BATCH_SIZE:
        raise ValueError ( f"Inference batches must contain 1 to {INFERENCE_BATCH_SIZE} images" )
    try:
        output = session.run ( [ session.get_outputs () [ 0 ].name ], { session.get_inputs () [ 0 ].name: inputs } ) [ 0 ]
        if output.shape != ( batch_size, EMBEDDING_SIZE ) or output.dtype != float32:
            raise ModelError ( f"Model inference must return float32 [{batch_size}, 576] embeddings" )
        embeddings = stack ( [ _normalize ( row ) for row in output ] )
    except ModelError:
        raise
    except Exception as error:
        raise ModelError ( f"Unable to infer a finite nonzero embedding: {error}" ) from error

    # Return unit rows; malformed model data must never reach cosine clustering.

    return embeddings


#-----------------------------------------------------------------------------------------------------------------------
# Function: embed
#
# Description:
#
#   Infer one pooled embedding through the same validation boundary as a full batch.
#
# Arguments:
#
#   image   : Nonempty oriented RGB image owned by the caller.
#   session : Reusable local CPU ONNX inference session.
#
# Returns:
#
#   One unit pooled content embedding.
#-----------------------------------------------------------------------------------------------------------------------

def embed ( image: Image, session: InferenceSession ) -> ndarray:

    # Infer one pooled embedding through the same validation boundary as a full batch.

    # Return data to caller.

    return embed_batch ( preprocess ( image ), session ) [ 0 ]


#-----------------------------------------------------------------------------------------------------------------------
# Function: prepare
#
# Description:
#
#   Detach bounded arrays while the caller owns just one full-resolution RGB image.
#
# Arguments:
#
#   image : Nonempty oriented RGB image owned by the caller.
#
# Returns:
#
#   Detached model input, color histogram, and duplicate evidence.
#-----------------------------------------------------------------------------------------------------------------------

def prepare ( image: Image ) -> PreparedImage:

    # Detach bounded arrays while the caller owns just one full-resolution RGB image.

    # Return data to caller.

    return PreparedImage (
        inputs       = preprocess ( image ),
        histogram    = histogram ( image ),
        dhash        = dhash ( image ),
        aspect_ratio = image.width / image.height,
        thumbnail    = verification_thumbnail ( image ),
    )


#-----------------------------------------------------------------------------------------------------------------------
# Function: extract_batch
#
# Description:
#
#   Consume a bounded collection of detached inputs, preserving input order and per-image duplicate evidence.
#
# Arguments:
#
#   prepared_images : Bounded sequence of detached image inputs and duplicate evidence.
#   session         : Reusable local CPU ONNX inference session.
#   alpha           : Content contribution in [0, 1]; color contributes 1 - alpha.
#
# Returns:
#
#   Feature records in the same order as the prepared input images.
#-----------------------------------------------------------------------------------------------------------------------

def extract_batch (
    prepared_images: Sequence [ PreparedImage ], session: InferenceSession, alpha: float = 0.80,
) -> tuple [ ImageFeatures, ... ]:

    # Consume a bounded collection of detached inputs, preserving input order and per-image duplicate evidence.

    if not 1 <= len ( prepared_images ) <= INFERENCE_BATCH_SIZE:
        raise ValueError ( f"Feature batches must contain 1 to {INFERENCE_BATCH_SIZE} images" )

    # Infer one bounded batch, then combine each embedding with its retained color and duplicate evidence.

    embeddings = embed_batch ( concatenate ( [ item.inputs for item in prepared_images ] ), session )

    # Return data to caller.

    return tuple (
        ImageFeatures (
            embedding    = embedding,
            histogram    = item.histogram,
            vector       = fuse ( embedding, item.histogram, alpha ),
            dhash        = item.dhash,
            aspect_ratio = item.aspect_ratio,
            thumbnail    = item.thumbnail,
        )
        for item, embedding in zip ( prepared_images, embeddings, strict = True )
    )


#-----------------------------------------------------------------------------------------------------------------------
# Function: extract
#
# Description:
#
#   Extract one compact feature record from an oriented RGB image without retaining or writing the original.
#
# Arguments:
#
#   image   : Nonempty oriented RGB image owned by the caller.
#   session : Reusable local CPU ONNX inference session.
#   alpha   : Content contribution in [0, 1]; color contributes 1 - alpha.
#
# Returns:
#
#   One detached feature record for the input image.
#-----------------------------------------------------------------------------------------------------------------------

def extract ( image: Image, session: InferenceSession, alpha: float = 0.80 ) -> ImageFeatures:

    # Extract one compact feature record from an oriented RGB image without retaining or writing the original.

    # Return data to caller.

    return extract_batch ( [ prepare ( image ) ], session, alpha ) [ 0 ]
