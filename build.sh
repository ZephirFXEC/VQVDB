#!/bin/bash

HELP=0
BUILDTYPE="Release"
RUNTESTS=0
REMOVEOLDDIR=0
EXPORTCOMPILECOMMANDS=0
VERSION="0.0.0"
INSTALL=0
INSTALLDIR="$PWD/install"
HOUPATH=""

# Parse command line arguments
parse_args()
{
    [ "$1" == "--help" ] && HELP=1
    [ "$1" == "-h" ] && HELP=1

    [ "$1" == "--debug" ] && BUILDTYPE="Debug"

    [ "$1" == "--reldebug" ] && BUILDTYPE="RelWithDebInfo"

    [ "$1" == "--tests" ] && RUNTESTS=1

    [ "$1" == "--clean" ] && REMOVEOLDDIR=1

    [ "$1" == "--install" ] && INSTALL=1

    [ "$1" == "--export-compile-commands" ] && EXPORTCOMPILECOMMANDS=1

    [ "$1" == *"version"* ] && parse_version $1

    [ "$1" == *"installdir"* ] && parse_install_dir $1

    [ "$1" == *"houdinipath"* ] && parse_houdini_path $1
}

# Parse the version from a command line argument
parse_version()
{
    VERSION="$( cut -d ':' -f 2- <<< "$1" )"
    log_info "Version specified by user: $VERSION"
}

# Parse the installation dir from a command line argument
parse_install_dir()
{
    INSTALLDIR="$( cut -d ':' -f 2- <<< "$1" )"
    log_info "Install directory specified by user: $INSTALLDIR"
}

# Parse the houdini path from a command line argument
parse_houdini_path()
{
    HOUPATH="$( cut -d ':' -f 2- <<< "$1" )"
    log_info "Houdini path specified by user: $HOUPATH"
}

# Log an information message to the console
log_info()
{
    echo "[INFO] : $1"
}

# Log a warning message to the console
log_warning()
{
    echo "[WARNING] : $1"
}

# Log an error message to the console
log_error()
{
    echo "[ERROR] : $1"
}

# Wrap to avoid command line args propagation
source_vcpkg_bootstrap()
{
    source bootstrap-vcpkg.sh
}

# Entry point

log_info "Building VQVDB"

if [[ -n "${VCPKG_ROOT}" ]]; then
    log_info "Using existing VCPKG_ROOT from environment: $VCPKG_ROOT"
    if [[ ! -x "$VCPKG_ROOT/vcpkg" ]]; then
        log_error "VCPKG_ROOT is set but vcpkg executable not found in it."
        exit 1
    fi
else
    if [[ -d "vcpkg" ]]; then
        export VCPKG_ROOT=$PWD/vcpkg
        log_info "Using local vcpkg directory: $VCPKG_ROOT"
    else
        log_info "Vcpkg can't be found, cloning and preparing it"
        git clone https://github.com/microsoft/vcpkg.git
        cd vcpkg
        source_vcpkg_bootstrap
        cd ..
        export VCPKG_ROOT=$PWD/vcpkg
    fi
fi

export CMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

if [[ $PATH != *$VCPKG_ROOT* ]]; then
    log_info "Can't find vcpkg root in PATH, appending it"
    export PATH=$PATH:$VCPKG_ROOT
fi

for arg in "$@"
do
    parse_args "$arg"
done

if [[ $HELP -eq 1 ]]; then
    echo "VQVDB build script for Linux"
    echo ""
    echo "Usage:"
    echo "  - build <arg1> <arg2> ..."
    echo "    - args:"
    echo "      --debug: builds in debug mode, defaults to release"
    echo "      --tests: runs CMake tests, if any"
    echo "      --clean: removes the old build directory"
    echo "      --install: runs CMake installation, if any"
    echo "      --installdir:<install_directory_path>: path to the install directory, default to ./install"
    echo "      --version:<version>: specifies the version, defaults to $VERSION"
    echo "      --houdinipath:<houdini_path>: specifies the path to HOUDINI"
    echo "      --help/-h: displays this message and exits"
    exit 0
fi

if [[ -z "${HOUPATH}" ]]; then
    if [[ -n "${HFS}" ]]; then
        log_info "HFS is already set, using it: $HFS"
    else
        log_error "Houdini path must be specified, either by setting HFS env var or by using --houdinipath:<path> arg"
        exit 1
    fi
else
    export HFS="${HOUPATH}"
fi

log_info "Build type: $BUILDTYPE"
log_info "Build version: $VERSION"

if [[ -d "build" && $REMOVEOLDDIR -eq 1 ]]; then
    log_info "Removing old build directory"
    rm -rf build
fi

if [[ -d "$INSTALLDIR" && $REMOVEOLDDIR -eq 1 ]]; then
    log_info "Removing old install directory"
    rm -rf "$INSTALLDIR"
fi

cmake -S . -B build -T v142 -DRUN_TESTS=$RUNTESTS -DCMAKE_EXPORT_COMPILE_COMMANDS=$EXPORTCOMPILECOMMANDS -DCMAKE_BUILD_TYPE=$BUILDTYPE -DVERSION=$VERSION

if [[ $? -ne 0 ]]; then
    log_error "Error during CMake configuration"
    exit 1 
fi

cd build

cmake --build . -- -j $(nproc)

if [[ $? -ne 0 ]]; then
    log_error "Error during CMake build"
    cd ..
    exit 1
fi

if [[ $RUNTESTS -eq 1 ]]; then
    ctest --output-on-failure -C $BUILDTYPE
    
    if [[ $? -ne 0 ]]; then
        log_error "Error during CMake testing"
        cd ..
        exit 1
    fi
fi

if [[ $INSTALL -eq 1 ]]; then
    cmake --install . --config $BUILDTYPE --prefix $INSTALLDIR
    
    if [[ $? -ne 0 ]]; then
        log_error "Error during CMake installation"
        cd ..
        exit 1
    fi
fi

cd ..

if [[ $EXPORTCOMPILECOMMANDS -eq 1 ]]; then
    cp ./build/compile_commands.json ./compile_commands.json
    log_info "Copied compile_commands.json to root directory"
fi