cmake_minimum_required(VERSION 3.31)
project(VQVDB LANGUAGES CUDA CXX)

# ------- Find CUDA -----------
find_package(CUDAToolkit REQUIRED)
find_package(Caffe2 CONFIG REQUIRED)
find_package(Torch CONFIG REQUIRED)

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)


# Set vars
if (UNIX)
    set(openvdb_lib "$ENV{HFS}/dsolib/libopenvdb_sesi.so")
endif ()

if (WIN32)
    set(openvdb_lib "$ENV{HFS}/custom/houdini/dsolib/openvdb_sesi.lib")
endif ()

# ------- Find Houdini --------
# Points CMake to $HFS/toolkit/cmake
list(APPEND CMAKE_PREFIX_PATH "$ENV{HFS}/toolkit/cmake")
find_package(Houdini REQUIRED)

# Ccreate an interface library capturing Houdini & openvdb_sesi
add_library(HoudiniVDB INTERFACE)
target_link_libraries(HoudiniVDB INTERFACE Houdini ${openvdb_lib})
target_include_directories(HoudiniVDB INTERFACE
        $ENV{HFS}/toolkit/include
)


add_library(kernels Kernel.cu)
target_link_libraries(kernels CUDA::cudart)

add_executable(VQVDB
        LeafExtractor.cpp
        LeafExtractor.hpp
        VQVAE_Decoder.cpp)

target_link_libraries(VQVDB HoudiniVDB torch protobuf::libprotobuf kernels)

list(APPEND CMAKE_PREFIX_PATH "libtorch")
