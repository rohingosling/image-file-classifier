#-----------------------------------------------------------------------------------------------------------------------
# Program: verify_executable.py
# Project: classifyimages
# Version: 1.0.0
# Date:    2026-10-03
# Author:  Rohin Gosling
#
# Description:
#
#   Audit the release payload and exercise image conversion from a directory containing only the executable.
#-----------------------------------------------------------------------------------------------------------------------

from argparse   import ArgumentParser
from csv        import DictReader
from fnmatch    import fnmatchcase
from hashlib    import sha256
from os         import environ
from pathlib    import Path
from shutil     import copy2
from struct     import unpack_from
from subprocess import run
from tempfile   import TemporaryDirectory

from PIL                         import Image
from PyInstaller.archive.readers import CArchiveReader, pkg_archive_contents


#-----------------------------------------------------------------------------------------------------------------------
# Function: require
#
# Description:
#
#   Stop verification when a release invariant fails.
#-----------------------------------------------------------------------------------------------------------------------

def require ( condition: bool, message: str ) -> None:

    if not condition:
        raise RuntimeError ( message )


#-----------------------------------------------------------------------------------------------------------------------
# Function: audit_payload
#
# Description:
#
#   Require an x64 console executable with its model, interpreter, native dependencies, and no export-only frameworks.
#-----------------------------------------------------------------------------------------------------------------------

def audit_payload ( executable: Path, model: Path | None ) -> None:

    executable_bytes = executable.read_bytes ()
    header_offset    = unpack_from ( "<I", executable_bytes, 0x3C ) [ 0 ]
    require ( executable_bytes [ header_offset : header_offset + 4 ] == b"PE\0\0", "Expected a Windows PE executable" )
    require ( unpack_from ( "<H", executable_bytes, header_offset + 4 ) [ 0 ] == 0x8664, "Expected an x64 executable" )
    require ( unpack_from ( "<H", executable_bytes, header_offset + 24 + 68 ) [ 0 ] == 3, "Expected a console executable" )
    archive    = CArchiveReader ( str ( executable ) )
    entries    = { name.replace ( "\\", "/" ).lower (): name for name in archive.toc }
    model_name = "models/mobilenet_v3_small.onnx"
    require ( model_name in entries, "The executable does not contain its default ONNX model" )
    embedded_model_hash = sha256 ( archive.extract ( entries [ model_name ] ) ).hexdigest ()
    if model is not None:
        require ( embedded_model_hash == sha256 ( model.read_bytes () ).hexdigest (), "Embedded model differs from build input" )

    # Check DLL and extension presence independently of libraries installed on the runner.

    required_patterns = (
        "python312.dll", "vcruntime140.dll", "vcruntime140_1.dll",
        "onnxruntime/capi/onnxruntime.dll", "onnxruntime/capi/onnxruntime_providers_shared.dll",
        "onnxruntime/capi/onnxruntime_pybind11_state*.pyd", "pil/_imaging*.pyd", "pil/_webp*.pyd",
        "numpy/_core/_multiarray_umath*.pyd", "sklearn/cluster/_hierarchical_fast*.pyd", "scipy/*.pyd",
    )
    for pattern in required_patterns:
        require ( any ( fnmatchcase ( name, pattern ) for name in entries ), f"Required payload entry missing: {pattern}" )
    forbidden_packages = { "torch", "torchvision", "tensorflow", "transformers" }
    contents           = pkg_archive_contents ( str ( executable ) )
    require ( not any ( name.replace ( "\\", "/" ).split ( "/" ) [ 0 ].split ( "." ) [ 0 ] in forbidden_packages for name in contents ),
              "An export-only framework leaked into the executable" )
    print ( f"Payload verified: x64 console, bundled interpreter/native libraries, model SHA-256 {embedded_model_hash}" )
    print ( f"Executable SHA-256: {sha256(executable_bytes).hexdigest()}" )


#-----------------------------------------------------------------------------------------------------------------------
# Function: create_fixtures
#
# Description:
#
#   Create independent image groups, an exact duplicate, transparency, EXIF orientation, and skipped-file candidates.
#-----------------------------------------------------------------------------------------------------------------------

def create_fixtures ( source: Path ) -> dict [ str, tuple [ int, int ] ]:

    source.mkdir ()
    with Image.new ( "RGB", ( 64, 40 ), ( 0, 0, 255 ) ) as image:
        image.save ( source / "blue.png" )
    copy2 ( source / "blue.png", source / "blue-copy.png" )
    with Image.new ( "RGB", ( 91, 57 ), ( 255, 255, 0 ) ) as image:
        orientation         = Image.Exif ()
        orientation [ 274 ] = 6
        image.save ( source / "yellow.jpg", quality = 95, exif = orientation )
    with Image.new ( "RGB", ( 37, 61 ), ( 0, 255, 0 ) ) as image:
        image.save ( source / "green.webp", lossless = True )
    with Image.new ( "RGBA", ( 79, 43 ), ( 255, 0, 0, 128 ) ) as image:
        image.save ( source / "transparent.png" )
    ( source / "broken.png" ).write_bytes ( b"not a PNG" )
    ( source / "notes.txt" ).write_text ( "Unsupported extension.\n", encoding = "utf-8" )

    # Return expected full-resolution dimensions after EXIF transpose.

    return { "blue.png": ( 64, 40 ), "blue-copy.png": ( 64, 40 ), "yellow.jpg": ( 57, 91 ),
             "green.webp": ( 37, 61 ), "transparent.png": ( 79, 43 ) }


#-----------------------------------------------------------------------------------------------------------------------
# Function: verify_runtime
#
# Description:
#
#   Run the copied executable without Python directories on PATH and inspect every image and manifest row.
#-----------------------------------------------------------------------------------------------------------------------

def verify_runtime ( executable: Path, work_directory: Path ) -> None:

    binary_directory = work_directory / "binary"
    source           = work_directory / "source"
    destination      = work_directory / "output"
    binary_directory.mkdir ()
    isolated_executable = binary_directory / "imageclassifier.exe"
    copy2 ( executable, isolated_executable )
    expected_sizes  = create_fixtures ( source )
    original_hashes = { path.name: sha256 ( path.read_bytes () ).hexdigest () for path in source.iterdir () }
    environment     = { key: value for key, value in environ.items () if not key.upper ().startswith ( "PYTHON" ) }
    environment.pop ( "VIRTUAL_ENV", None )
    environment [ "PATH" ] = str ( Path ( environ [ "SystemRoot" ] ) / "System32" ) + ";" + environ [ "SystemRoot" ]
    require ( [ path.name for path in binary_directory.iterdir () ] == [ "imageclassifier.exe" ], "Executable is not isolated" )

    # Help alone skips model and image imports, so also perform actual multi-group classification.

    for arguments in ( [ "--help" ], [ "-s", str ( source ), "-d", str ( destination ), "--threshold", "0" ] ):
        process = run ( [ str ( isolated_executable ), *arguments ], cwd = binary_directory, env = environment,
                        capture_output = True, encoding = "utf-8", errors = "replace", timeout = 180 )
        print ( process.stdout, end = "" )
        require ( process.returncode == 0,
                  f"Frozen command failed with exit {process.returncode}: {process.stderr}" )
        if arguments == [ "--help" ]:
            required_options = ( "--source", "--dest", "--threshold", "--color-weight", "--alpha",
                                 "--recursive", "--manifest", "--model" )
            require ( all ( option in process.stdout for option in required_options ), "Frozen help is missing CLI options" )
    with ( destination / "manifest.csv" ).open ( newline = "", encoding = "utf-8" ) as manifest_file:
        rows = list ( DictReader ( manifest_file ) )
    successful_rows = { row [ "source_path" ]: row for row in rows if not row [ "skipped_reason" ] }
    skipped_paths   = { row [ "source_path" ] for row in rows if row [ "skipped_reason" ] }
    require ( len ( rows ) == 7 and set ( successful_rows ) == set ( expected_sizes ), "Manifest has missing or repeated images" )
    require ( skipped_paths == { "broken.png", "notes.txt" }, "Manifest is missing skipped files" )
    output_names = set ()
    for source_name, expected_size in expected_sizes.items ():
        row = successful_rows [ source_name ]
        require ( row [ "output_name" ] == f"{int(row['cluster']):04d}-{int(row['index']):08d}.png", "Invalid output name" )
        require ( int ( row [ "cluster" ] ) > 0 and int ( row [ "index" ] ) > 0, "Output indices must be positive" )
        output_names.add ( row [ "output_name" ] )
        with Image.open ( destination / row [ "output_name" ] ) as converted:
            converted.load ()
            require ( converted.format == "PNG" and converted.mode == "RGB" and converted.size == expected_size,
                      f"Invalid full-resolution RGB PNG: {source_name}" )
            if source_name == "transparent.png":
                require ( converted.getpixel ( ( 0, 0 ) ) == ( 255, 127, 127 ), "Alpha was not composited onto white" )
    require ( len ( output_names ) == 5, "Output filenames are not unique" )
    require ( { path.name for path in destination.iterdir () } == output_names | { "manifest.csv" }, "Unexpected output files" )
    require ( successful_rows [ "blue.png" ] [ "cluster" ] == successful_rows [ "blue-copy.png" ] [ "cluster" ],
              "Exact duplicates were split" )
    require ( len ( { row [ "cluster" ] for row in successful_rows.values () } ) == 4,
              "Threshold zero did not preserve four independent image groups" )
    require ( original_hashes == { path.name: sha256 ( path.read_bytes () ).hexdigest () for path in source.iterdir () },
              "Source files changed" )
    require ( [ path.name for path in binary_directory.iterdir () ] == [ "imageclassifier.exe" ], "Adjacent files were created" )
    print ( "Frozen smoke passed: 5 images, 4 clusters, 2 skips; JPEG/PNG/WebP, EXIF, alpha, naming, and source integrity." )
    print ( "Scope: executable-only directory with restricted PATH; host Python and network access remain available." )


#-----------------------------------------------------------------------------------------------------------------------
# Function: main
#
# Description:
#
#   Audit and smoke-test the supplied release executable, optionally comparing its embedded model with the build input.
#-----------------------------------------------------------------------------------------------------------------------

def main () -> None:

    parser = ArgumentParser ( description = "Verify the self-contained Windows release executable." )
    parser.add_argument ( "executable", type = Path )
    parser.add_argument ( "--model", type = Path, help = "Vendored model to compare with the embedded payload" )
    arguments  = parser.parse_args ()
    executable = arguments.executable.resolve ( strict = True )
    model      = arguments.model.resolve ( strict = True ) if arguments.model is not None else None
    audit_payload ( executable, model )
    with TemporaryDirectory ( prefix = "imageclassifier-release-" ) as temporary_directory:
        verify_runtime ( executable, Path ( temporary_directory ) )


if __name__ == "__main__":
    main ()
