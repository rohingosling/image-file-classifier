#-----------------------------------------------------------------------------------------------------------------------
# Module:  cluster.py
# Project: classifyimages
# Version: 1.0.0
# Date:    2023-03-01
# Author:  Rohin Gosling
#
# Description:
#
#   Confirm complete-link duplicate groups, cluster their representatives, and assign deterministic PNG names.
#-----------------------------------------------------------------------------------------------------------------------

from collections.abc import Callable, Sequence
from dataclasses     import dataclass
from math            import isfinite as finite_scalar
from numbers         import Integral
from pathlib         import PureWindowsPath

from numpy           import absolute, asarray, clip, eye, float64, isfinite, ndarray, stack, subtract
from numpy.linalg    import norm
from sklearn.cluster import AgglomerativeClustering

# Output-name bounds and duplicate-confirmation limits.

MAX_CLUSTERS           = 9_999
MAX_IMAGES_PER_CLUSTER = 99_999_999
MAX_DHASH_DISTANCE     = 4
MAX_ASPECT_RATIO       = 1.05
MAX_THUMBNAIL_ERROR    = 0.08


#-----------------------------------------------------------------------------------------------------------------------
# Class: ClusteringError
#
# Description:
#
#   Invalid clustering input or an unrepresentable output plan; no files have been written.
#-----------------------------------------------------------------------------------------------------------------------

class ClusteringError ( ValueError ):

    pass


#-----------------------------------------------------------------------------------------------------------------------
# Class: NamedImage
#
# Description:
#
#   One planned output, retaining the original feature-row index for the output writer.
#
# Attributes:
#
#   image_index          : Original feature-row index identifying the source image.
#   relative_path        : Normalized source-relative path.
#   cluster              : Dense one-based cluster identifier.
#   index                : One-based member index within the cluster.
#   output_name          : Fixed-width PNG basename.
#   distance_to_centroid : Cosine distance to the expanded cluster centroid or medoid fallback.
#-----------------------------------------------------------------------------------------------------------------------

@dataclass ( frozen = True )
class NamedImage:

    image_index:          int
    relative_path:        str
    cluster:              int
    index:                int
    output_name:          str
    distance_to_centroid: float


#-----------------------------------------------------------------------------------------------------------------------
# Function: _normalized_paths
#
# Description:
#
#   Canonicalize Windows or forward-slash relative paths without accessing the filesystem.
#
# Arguments:
#
#   relative_paths : Source-relative paths corresponding to the feature rows.
#
# Returns:
#
#   Case-preserving forward-slash relative paths in input order.
#-----------------------------------------------------------------------------------------------------------------------

def _normalized_paths ( relative_paths: Sequence [ str ] ) -> tuple [ str, ... ]:

    # Canonicalize Windows or forward-slash relative paths without accessing the filesystem.

    paths = []
    for relative_path in relative_paths:
        path = PureWindowsPath ( relative_path )
        if path.drive or path.root or not path.parts or ".." in path.parts:
            raise ClusteringError ( "Image paths must be nonempty source-relative paths without '..'" )
        paths.append ( path.as_posix () )
    if len ( set ( paths ) ) != len ( paths ):
        raise ClusteringError ( "Image paths must be unique after separator normalization" )

    # Return case-preserving paths for every ordering and tie-break decision.

    return tuple ( paths )


#-----------------------------------------------------------------------------------------------------------------------
# Function: _unit_vectors
#
# Description:
#
#   Validate and normalize feature rows in float64 without modifying the caller's arrays.
#
# Arguments:
#
#   vectors     : One feature vector per input image.
#   image_count : Number of input images expected in the feature matrix.
#
# Returns:
#
#   Detached float64 unit feature rows.
#-----------------------------------------------------------------------------------------------------------------------

def _unit_vectors ( vectors: ndarray, image_count: int ) -> ndarray:

    # Validate and normalize feature rows in float64 without modifying the caller's arrays.

    values = asarray ( vectors, dtype = float64 )
    if values.ndim != 2 or values.shape [ 0 ] != image_count or values.shape [ 1 ] == 0 or image_count == 0:
        raise ClusteringError ( "Expected one nonempty feature vector per image and at least one image" )
    if not isfinite ( values ).all ():
        raise ClusteringError ( "Feature vectors must be finite" )
    magnitudes = norm ( values, axis = 1 )
    if not isfinite ( magnitudes ).all () or ( magnitudes == 0 ).any ():
        raise ClusteringError ( "Feature vectors must have finite nonzero L2 norms" )

    # Return unit rows so cosine ordering and means use the same geometry as clustering.

    return values / magnitudes [ :, None ]


#-----------------------------------------------------------------------------------------------------------------------
# Function: _partition
#
# Description:
#
#   Require each image exactly once and canonicalize both groups and members by relative path.
#
# Arguments:
#
#   groups : Groups of original image-row indices.
#   paths  : Normalized source-relative paths in input-row order.
#
# Returns:
#
#   Complete partition with groups and members sorted by relative path.
#-----------------------------------------------------------------------------------------------------------------------

def _partition ( groups: Sequence [ Sequence [ int ] ], paths: tuple [ str, ... ] ) -> tuple [ tuple [ int, ... ], ... ]:

    # Require each image exactly once and canonicalize both groups and members by relative path.

    seen           = set ()
    ordered_groups = []
    for members in groups:
        if len ( members ) == 0:
            raise ClusteringError ( "Image groups must not be empty" )
        for member in members:
            if isinstance ( member, bool ) or not isinstance ( member, Integral ) or not 0 <= member < len ( paths ):
                raise ClusteringError ( "Group members must be valid image indices" )
            if member in seen:
                raise ClusteringError ( "Every image must appear in exactly one group" )
            seen.add ( member )
        ordered_groups.append ( tuple ( sorted ( members, key = paths.__getitem__ ) ) )
    if len ( seen ) != len ( paths ):
        raise ClusteringError ( "Every image must appear in exactly one group" )

    # Canonical input order also makes scikit-learn's handling of equal distances repeatable.

    return tuple ( sorted ( ordered_groups, key = lambda members: paths [ members [ 0 ] ] ) )


#-----------------------------------------------------------------------------------------------------------------------
# Function: _check_cluster_count
#
# Description:
#
#   Reject names wider than four digits before returning any output plan.
#
# Arguments:
#
#   cluster_count : Number of output clusters to validate.
#
# Returns:
#
#   None.
#-----------------------------------------------------------------------------------------------------------------------

def _check_cluster_count ( cluster_count: int ) -> None:

    # Reject names wider than four digits before returning any output plan.

    if cluster_count > MAX_CLUSTERS:
        raise ClusteringError ( f"Result has {cluster_count:,} clusters; at most {MAX_CLUSTERS:,} fit four-digit names" )


#-----------------------------------------------------------------------------------------------------------------------
# Function: duplicates
#
# Description:
#
#   Return complete-link duplicate groups as indices into the original input rows.
#
#   Evidence comes from features.extract: 64-bit hashes, oriented width/height ratios, and white-padded float thumbnails
#   with shape [images, 64, 64, 3] in [0, 1]. Candidate pairs are processed in path order. A boolean compatibility
#   matrix is intersected on each merge, preserving every cross-pair guard.
#
# Arguments:
#
#   relative_paths : Source-relative paths corresponding to the feature rows.
#   dhashes        : Unsigned 64-bit difference hashes for the input images.
#   aspect_ratios  : Oriented width-to-height ratios for the input images.
#   thumbnails     : White-padded RGB thumbnails with shape [images, 64, 64, 3] in [0, 1].
#   progress       : Optional callback receiving completed and total work counts.
#
# Returns:
#
#   Complete-link duplicate groups containing original image-row indices.
#-----------------------------------------------------------------------------------------------------------------------

def duplicates (
    relative_paths: Sequence [ str ],
    dhashes: Sequence [ int ],
    aspect_ratios: ndarray,
    thumbnails: ndarray,
    *,
    progress: Callable [ [ int, int ], None ] | None = None,
) -> tuple [ tuple [ int, ... ], ... ]:

    # Return complete-link duplicate groups as indices into the original input rows.

    paths       = _normalized_paths ( relative_paths )
    image_count = len ( paths )
    ratios      = asarray ( aspect_ratios, dtype = float64 )
    pixels      = asarray ( thumbnails )
    if len ( dhashes ) != image_count or ratios.shape != ( image_count, ):
        raise ClusteringError ( "Expected one dHash and aspect ratio per image" )
    if any ( isinstance ( value, bool ) or not isinstance ( value, Integral ) or not 0 <= value < 1 << 64 for value in dhashes ):
        raise ClusteringError ( "dHash values must be unsigned 64-bit integers" )
    if not isfinite ( ratios ).all () or ( ratios <= 0 ).any ():
        raise ClusteringError ( "Aspect ratios must be finite and positive" )
    if pixels.shape != ( image_count, 64, 64, 3 ) or pixels.dtype.kind not in "fiu":
        raise ClusteringError ( "Expected numeric RGB thumbnails with shape [images, 64, 64, 3]" )
    if not isfinite ( pixels ).all () or ( pixels < 0 ).any () or ( pixels > 1 ).any ():
        raise ClusteringError ( "RGB thumbnails must contain finite values in [0, 1]" )
    ordered_indices = sorted ( range ( image_count ), key = paths.__getitem__ )
    compatible      = eye ( image_count, dtype = bool )

    # Compute each expensive RGB comparison once, after both cheap candidate guards pass.

    total_comparisons = image_count * ( image_count - 1 )
    completed         = 0
    for i, left_index in enumerate ( ordered_indices ):
        for j in range ( i + 1, image_count ):
            right_index = ordered_indices [ j ]
            if ( int ( dhashes [ left_index ] ) ^ int ( dhashes [ right_index ] ) ).bit_count () > MAX_DHASH_DISTANCE:
                continue
            if max ( ratios [ left_index ], ratios [ right_index ] ) / min ( ratios [ left_index ], ratios [ right_index ] ) > MAX_ASPECT_RATIO:
                continue
            error               = absolute ( subtract ( pixels [ left_index ], pixels [ right_index ], dtype = float64 ) ).mean ()
            compatible [ i, j ] = compatible [ j, i ] = error <= MAX_THUMBNAIL_ERROR
        completed += image_count - i - 1
        if progress is not None:
            progress ( completed, total_comparisons )

    # Merge compatible groups in canonical path order while tracking each member's current owner.

    owners           = list ( range ( image_count ) )
    members_by_owner = { i: [ i ] for i in range ( image_count ) }
    for i in range ( image_count ):
        for j in range ( i + 1, image_count ):
            left_owner, right_owner = sorted ( ( owners [ i ], owners [ j ] ) )
            if left_owner == right_owner or not compatible [ left_owner, right_owner ]:
                continue

            # Intersection requires every member of each current group to match the other group.

            compatible [ left_owner ] &= compatible [ right_owner ]
            compatible [ :, left_owner ]          = compatible [ left_owner ]
            compatible [ left_owner, left_owner ] = True
            right_members                         = members_by_owner.pop ( right_owner )
            members_by_owner [ left_owner ].extend ( right_members )
            for member in right_members:
                owners [ member ] = left_owner
        completed += image_count - i - 1
        if progress is not None:
            progress ( completed, total_comparisons )

    # Translate canonical row positions back to the caller's rows and retain path order.

    return tuple (
        tuple ( ordered_indices [ member ] for member in sorted ( members ) )
        for _, members in sorted ( members_by_owner.items () )
    )


#-----------------------------------------------------------------------------------------------------------------------
# Function: group
#
# Description:
#
#   Cluster unit mean duplicate representatives, expand membership, and order clusters by size then path.
#
#   Omitted duplicate_groups treats every image as a singleton. The cut is the scikit-learn average-linkage cosine cut:
#   merges at or above threshold are excluded. Each duplicate group has one vote, regardless of its image count. A
#   single group bypasses agglomeration. No destination or source file is accessed.
#
# Arguments:
#
#   vectors          : One feature vector per input image.
#   relative_paths   : Source-relative paths corresponding to the feature rows.
#   duplicate_groups : Complete-link duplicate groups, or None to use singleton groups.
#   threshold        : Nonnegative average-linkage cosine distance cut.
#
# Returns:
#
#   Expanded clusters sorted by decreasing image count, then relative path.
#-----------------------------------------------------------------------------------------------------------------------

def group (
    vectors: ndarray,
    relative_paths: Sequence [ str ],
    duplicate_groups: Sequence [ Sequence [ int ] ] | None = None,
    threshold: float                                       = 0.35,
) -> tuple [ tuple [ int, ... ], ... ]:

    # Cluster unit mean duplicate representatives, expand membership, and order clusters by size then path.

    paths  = _normalized_paths ( relative_paths )
    values = _unit_vectors ( vectors, len ( paths ) )
    if not finite_scalar ( threshold ) or threshold < 0:
        raise ClusteringError ( "threshold must be a finite nonnegative number" )
    if duplicate_groups is None:
        duplicate_groups = tuple ( ( i, ) for i in range ( len ( paths ) ) )
    ordered_groups = _partition ( duplicate_groups, paths )
    if len ( ordered_groups ) == 1:

        # Return data to caller.

        return ordered_groups

    # Give each duplicate group one unit mean representative, independent of its number of images.

    representatives = []
    for members in ordered_groups:
        mean      = values [ list ( members ) ].mean ( axis = 0 )
        magnitude = norm ( mean )
        if magnitude == 0:
            raise ClusteringError ( f"Duplicate group at '{paths[members[0]]}' has a zero mean; cosine representative is undefined" )
        representatives.append ( mean / magnitude )

    # Apply the average-linkage cosine cut to the representatives.

    labels = AgglomerativeClustering (
        n_clusters         = None,
        metric             = "cosine",
        linkage            = "average",
        distance_threshold = threshold,
    ).fit_predict ( stack ( representatives ) )

    # Expand representative labels back into all original image memberships.

    expanded = {}
    for label, members in zip ( labels, ordered_groups, strict = True ):
        expanded.setdefault ( int ( label ), [] ).extend ( members )
    _check_cluster_count ( len ( expanded ) )
    clusters = [ tuple ( sorted ( members, key = paths.__getitem__ ) ) for members in expanded.values () ]

    # Dense identifiers follow expanded image counts, never the arbitrary labels returned by scikit-learn.

    return tuple ( sorted ( clusters, key = lambda members: ( -len ( members ), paths [ members [ 0 ] ] ) ) )


#-----------------------------------------------------------------------------------------------------------------------
# Function: format_name
#
# Description:
#
#   Format a positive cluster/index pair without allowing either field to overflow its fixed width.
#
# Arguments:
#
#   cluster : Positive cluster identifier.
#   index   : Positive image index within the cluster.
#
# Returns:
#
#   Fixed-width PNG basename using four cluster digits and eight index digits.
#-----------------------------------------------------------------------------------------------------------------------

def format_name ( cluster: int, index: int ) -> str:

    # Format a positive cluster/index pair without allowing either field to overflow its fixed width.

    for value, maximum, label in ( ( cluster, MAX_CLUSTERS, "cluster" ), ( index, MAX_IMAGES_PER_CLUSTER, "index" ) ):
        if isinstance ( value, bool ) or not isinstance ( value, Integral ) or not 1 <= value <= maximum:
            raise ClusteringError ( f"{label} must be an integer from 1 to {maximum:,}" )

    # Return the final basename, independent of source extensions or filename prefixes.

    return f"{cluster:04d}-{index:08d}.png"


#-----------------------------------------------------------------------------------------------------------------------
# Function: _ordering_reference
#
# Description:
#
#   Use the expanded image centroid, or the path-first medoid when the mean has exactly zero norm.
#
# Arguments:
#
#   values : Unit feature rows belonging to one expanded cluster.
#   paths  : Normalized source-relative paths in input-row order.
#
# Returns:
#
#   Unit centroid or path-first medoid when the centroid has zero norm.
#-----------------------------------------------------------------------------------------------------------------------

def _ordering_reference ( values: ndarray, paths: tuple [ str, ... ] ) -> ndarray:

    # Use the expanded image centroid, or the path-first medoid when the mean has exactly zero norm.

    mean      = values.mean ( axis = 0 )
    magnitude = norm ( mean )
    if magnitude != 0:

        # Return data to caller.

        return mean / magnitude

    # Score one candidate at a time so the exceptional fallback needs no square distance allocation.

    medoid = min (
        range ( len ( paths ) ),
        key = lambda i: ( float ( clip ( 1.0 - values @ values [ i ], 0.0, 2.0 ).sum () ), paths [ i ] ),
    )

    # Return the actual member vector so even cancelling clusters have finite ordering distances.

    return values [ medoid ]


#-----------------------------------------------------------------------------------------------------------------------
# Function: assign_names
#
# Description:
#
#   Return a complete output plan sorted by dense cluster id and one-based member index.
#
#   distance_to_centroid is cosine distance to the expanded cluster's unit mean; for a cancelling mean it records
#   distance to the medoid fallback. Input membership and vectors are never modified.
#
# Arguments:
#
#   vectors        : One feature vector per input image.
#   relative_paths : Source-relative paths corresponding to the feature rows.
#   clusters       : Cluster memberships expressed as original image-row indices.
#   progress       : Optional callback receiving completed and total work counts.
#
# Returns:
#
#   Complete output plan ordered by cluster identifier and member index.
#-----------------------------------------------------------------------------------------------------------------------

def assign_names (
    vectors: ndarray,
    relative_paths: Sequence [ str ],
    clusters: Sequence [ Sequence [ int ] ],
    *,
    progress: Callable [ [ int, int ], None ] | None = None,
) -> tuple [ NamedImage, ... ]:

    # Return a complete output plan sorted by dense cluster id and one-based member index.

    paths = _normalized_paths ( relative_paths )
    _check_cluster_count ( len ( clusters ) )
    values           = _unit_vectors ( vectors, len ( paths ) )
    ordered_clusters = sorted ( _partition ( clusters, paths ), key = lambda members: ( -len ( members ), paths [ members [ 0 ] ] ) )
    plan             = []
    for cluster, members in enumerate ( ordered_clusters, start = 1 ):
        member_values = values [ list ( members ) ]
        reference     = _ordering_reference ( member_values, tuple ( paths [ member ] for member in members ) )
        distances     = clip ( 1.0 - member_values @ reference, 0.0, 2.0 )
        positions     = sorted ( range ( len ( members ) ), key = lambda i: ( float ( distances [ i ] ), paths [ members [ i ] ] ) )
        for index, position in enumerate ( positions, start = 1 ):
            image_index = members [ position ]
            plan.append ( NamedImage (
                image_index          = image_index,
                relative_path        = paths [ image_index ],
                cluster              = cluster,
                index                = index,
                output_name          = format_name ( cluster, index ),
                distance_to_centroid = float ( distances [ position ] ),
            ) )
        if progress is not None:
            progress ( cluster, len ( ordered_clusters ) )

    # Return only after all names are valid, leaving destination checks and writes to the output boundary.

    return tuple ( plan )
