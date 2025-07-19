# Building VQVDB

Since VQVDB relies on Torch, we use vcpkg as a package manager to ease and simplify the build process. We recommand using the latest version of libtorch, compatible with the cuda version you have installed. You can find the latest version of libtorch at https://pytorch.org/get-started/locally/.

You need to have CUDA Toolkit installed on your computer, along with CUDNN. If you have an error when compiling Nvidia CUTLASS, see https://github.com/microsoft/vcpkg/issues/43081.

Once every dependency has been built, VQVDB will look for Houdini to build the SOP nodes. You can specify a path to Houdini in the build scripts using `--houdinipath:<houdini_path>` or by setting the `HFS` environment variable.

To compile with Houdini toolkit you need to make sure that your MSVC version matches the one used by Houdini (for Houdini 20.5 is MSVC 2019 or v142). If CMake complains that it cannot find CUDA Toolkit, see https://github.com/abetlen/llama-cpp-python/issues/247.

You can use `--help` with the build scripts to see the different options available.

Happy encoding!