#-----------------------------------------------------------------------------------------------------------------------
# Module:  classifyimages.spec
# Project: classifyimages
# Version: 0.1.0
# Date:    2023-03-01
# Author:  Rohin Gosling
#
# Description:
#
#   Bundle the CPU application, native dependencies, and vendored ONNX model as one Windows console executable.
#-----------------------------------------------------------------------------------------------------------------------

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

# Resolve the bundled model before collecting application dependencies.

project_root = Path ( SPEC ).resolve ().parent
model_path   = project_root / "models" / "mobilenet_v3_small.onnx"
if not model_path.is_file ():
    raise FileNotFoundError ( f"Bundled ONNX model is missing: {model_path}" )

# Collect Python modules, package data, and native libraries required by the offline CPU runtime.

collected_data     = [ ( str ( model_path ), "models" ) ]
collected_binaries = []
hidden_imports     = []
for package_name in ( "PIL", "numpy", "sklearn", "onnxruntime" ):
    package_data, package_binaries, package_hidden_imports = collect_all ( package_name )
    collected_data.extend ( package_data )
    collected_binaries.extend ( package_binaries )
    hidden_imports.extend ( package_hidden_imports )

# Analyze the console entry point while excluding frameworks used only for offline model export.

analysis = Analysis (
    [ str ( project_root / "src" / "classifyimages" / "__main__.py" ) ],
    pathex        = [ str ( project_root / "src" ) ],
    binaries      = collected_binaries,
    datas         = collected_data,
    hiddenimports = hidden_imports,
    hookspath     = [],
    hooksconfig   = {},
    runtime_hooks = [],
    excludes      = [ "torch", "torchvision", "tensorflow", "transformers" ],
    noarchive     = False,
)
archive = PYZ ( analysis.pure )

# Embed the archive, native libraries, package data, and model in one portable console executable.

executable = EXE (
    archive,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name                      = "imageclassifier",
    debug                     = False,
    bootloader_ignore_signals = False,
    strip                     = False,
    upx                       = False,
    console                   = True,
)
