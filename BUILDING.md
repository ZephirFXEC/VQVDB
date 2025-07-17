# Building VQVDB

Since VQVDB relies on Torch, we use vcpkg as a package manager to ease and simplify the build process. vcpkg will automatically be downloaded by build scripts (build.bat for Windows, build.sh for Linux) and dependencies will be built prior to building VQVDB.

You need to have CUDA Toolkit installed on your computer, along with CUDNN. If you have an error when compiling Nvidia CUTLASS, see https://github.com/microsoft/vcpkg/issues/43081.

Once every dependency has been built, VQVDB will look for Houdini to build the SOP nodes. You can specify a path to Houdini in the build scripts using `--houdinipath:<houdini_path>` or by setting the `HFS` environment variable.

You can use `--help` with the build scripts to see the different options available.

Happy encoding!