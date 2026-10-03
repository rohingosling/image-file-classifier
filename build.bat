@echo off
rem --------------------------------------------------------------------------------------------------------------------
rem Program: classifyimages build
rem Version: 1.0.0
rem Date:    2023-03-01
rem Author:  Rohin Gosling
rem
rem Description:
rem
rem   Build the portable console executable with the project environment and bundled ONNX model.
rem --------------------------------------------------------------------------------------------------------------------

setlocal

pushd "%~dp0"
if errorlevel 1 goto :directory_error

set "PROJECT_PYTHON=.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" (
    echo Build environment missing. From this folder, run:
    echo     py -3.12 -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    goto :build_error
)

if not exist "models\mobilenet_v3_small.onnx" (
    echo The bundled model is missing: models\mobilenet_v3_small.onnx
    goto :build_error
)

echo Building imageclassifier.exe...
"%PROJECT_PYTHON%" -m PyInstaller --noconfirm --distpath "dist" --workpath "build\pyinstaller" "classifyimages.spec"
if errorlevel 1 goto :build_error

if not exist "dist\imageclassifier.exe" (
    echo The build finished without producing dist\imageclassifier.exe.
    goto :build_error
)

echo.
echo Build complete: "%CD%\dist\imageclassifier.exe"
popd
if not defined CLASSIFYIMAGES_SKIP_PAUSE pause
exit /b 0

:build_error
echo.
echo Build failed. Review the messages above.
popd
if not defined CLASSIFYIMAGES_SKIP_PAUSE pause
exit /b 1

:directory_error
echo Could not enter the project folder.
if not defined CLASSIFYIMAGES_SKIP_PAUSE pause
exit /b 1
