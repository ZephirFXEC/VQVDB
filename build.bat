@echo off
setlocal enabledelayedexpansion

rem Utility batch script to build the libraries and nodes
rem Entry point

set HELP=0
set BUILDTYPE=Release
set RUNTESTS=0
set REMOVEOLDDIR=0
set ARCH=x64
set VERSION="0.0.0"
set INSTALL=0
set INSTALLDIR=%CD%\install
set "HOUPATH="

for %%x in (%*) do (
    call :ParseArg %%~x
)

if %HELP% equ 1 (
    echo VQVDB build script for Windows
    echo.
    echo Usage:
    echo   - build ^<arg1^> ^<arg2^> ...
    echo     - args:
    echo       --debug: builds in debug mode, defaults to release
    echo       --tests: runs CMake tests, if any
    echo       --clean: removes the old build directory
    echo       --install: runs CMake installation, if any
    echo       --installdir:^<install_directory_path^>: path to the install directory, default to ./install
    echo       --version:^<version^>: specifies the version, defaults to %VERSION%
    echo       --houdinipath:^<houdini_path^>: specifies the path to HOUDINI
    echo       --help/-h: displays this message and exits

    exit /B 0
)

call :LogInfo "Building VQVDB"

if not exist vcpkg (
    call :LogInfo "Vcpkg can't be found, cloning and preparing it"
    git clone https://github.com/microsoft/vcpkg.git
    cd vcpkg
    call bootstrap-vcpkg.bat
    cd ..
)

set VCPKG_ROOT=%CD%/vcpkg
set CMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake

echo.!PATH! | findstr /C:"!VCPKG_ROOT!" 1>nul

if %errorlevel% equ 1 (
    call :LogInfo "Can't find vcpkg root in PATH, appending it"
    set PATH=!PATH!;!VCPKG_ROOT!
)

if %REMOVEOLDDIR% equ 1 (
    if exist build (
        call :LogInfo "Removing old build directory"
        rmdir /s /q build
    )

    if exist %INSTALLDIR% (
        call :LogInfo "Removing old install directory"
        rmdir /s /q %INSTALLDIR%
    )
)

if not defined HOUPATH (
    if defined HFS (
        call :LogInfo "HFS is already set, using it: %HFS%"
    ) else (
        call :LogError "Houdini path must be specified, either by setting HFS env var or by using --houdinipath:<path> arg"
        exit /B 1
    )
) else (
    set HFS=%HOUPATH%
)

call :LogInfo "Build type: %BUILDTYPE%"
call :LogInfo "Build version: %VERSION%"

cmake -S . -B build -DRUN_TESTS=%RUNTESTS% -A="%ARCH%" -DVERSION=%VERSION%

if %errorlevel% neq 0 (
    call :LogError "Error caught during CMake configuration"
    exit /B 1
)

cd build
cmake --build . --config %BUILDTYPE% -j %NUMBER_OF_PROCESSORS%

if %errorlevel% neq 0 (
    call :LogError "Error caught during CMake compilation"
    cd ..
    exit /B 1
)

if %RUNTESTS% equ 1 (
    ctest --output-on-failure -C %BUILDTYPE%

    if %errorlevel% neq 0 (
        call :LogError "Error caught during CMake testing"
        type build\Testing\Temporary\LastTest.log

        cd ..
        exit /B 1
    )
)

if %INSTALL% equ 1 (
    cmake --install . --config %BUILDTYPE% --prefix %INSTALLDIR%
    
    if %errorlevel% neq 0 (
        call :LogError "Error caught during CMake installation"
        cd ..
        exit /B 1
    )
)

cd ..

exit /B 0

rem //////////////////////////////////
rem Process args
:ParseArg

if "%~1" equ "--help" set HELP=1
if "%~1" equ "-h" set HELP=1

if "%~1" equ "--debug" set BUILDTYPE=Debug

if "%~1" equ "--reldebug" set BUILDTYPE=RelWithDebInfo

if "%~1" equ "--tests" set RUNTESTS=1

if "%~1" equ "--clean" set REMOVEOLDDIR=1

if "%~1" equ "--install" set INSTALL=1

if "%~1" equ "--export-compile-commands" (
    call :LogWarning "Exporting compile commands is not supported on Windows for now"
)

echo "%~1" | find /I "version">nul && (
    call :ParseVersion %~1
)

echo "%~1" | find /I "installdir">nul && (
    call :ParseInstallDir %~1
)

echo "%~1" | find /I "houdinipath">nul && (
    call :ParseHoudiniPath %~1
)

exit /B 0
rem //////////////////////////////////

rem //////////////////////////////////
rem Parse the version from the command line arg (ex: --version:0.1.3)
:ParseVersion

for /f "tokens=2 delims=:" %%a in ("%~1") do (
    set VERSION=%%a
    call :LogInfo "Version specified by the user: %%a"
)

exit /B 0
rem //////////////////////////////////

rem //////////////////////////////////
rem Parse the install dir from the command line 
:ParseInstallDir

for /f "tokens=1* delims=:" %%a in ("%~1") do (
    set INSTALLDIR=%%b
    call :LogInfo "Install directory specified by the user: %%b"
)

exit /B 0
rem //////////////////////////////////

rem //////////////////////////////////
rem Parse the houdini path from the command line 
:ParseHoudiniPath

for /f "tokens=1* delims=:" %%a in ("%~1") do (
    set HOUPATH=%%b
    call :LogInfo "Houdini path specified by the user: %%b"
)

exit /B 0
rem //////////////////////////////////

rem //////////////////////////////////
rem Log errors
:LogError

echo [ERROR] : %~1

exit /B 0
rem //////////////////////////////////

rem //////////////////////////////////
rem Log warnings 
:LogWarning

echo [WARNING] : %~1

exit /B 0
rem //////////////////////////////////

rem //////////////////////////////////
rem Log infos 
:LogInfo

echo [INFO] : %~1

exit /B 0
rem //////////////////////////////////