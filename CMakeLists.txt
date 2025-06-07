cmake_minimum_required(VERSION 3.28)
project(VQVDB LANGUAGES CUDA CXX)

set(USE_CUDNN 1)
# ------- Find CUDA -----------
find_package(CUDAToolkit REQUIRED)
find_package(Torch CONFIG REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)
find_package(pybind11 CONFIG)
find_package(TBB CONFIG REQUIRED)


set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Set vars for OpenVDB
if (UNIX)
    set(openvdb_lib "$ENV{HFS}/dsolib/libopenvdb_sesi.so")
endif ()

if (WIN32)
    set(openvdb_lib "$ENV{HFS}/custom/houdini/dsolib/openvdb_sesi.lib")
endif ()

# ------- Find Houdini --------
list(APPEND CMAKE_PREFIX_PATH "$ENV{HFS}/toolkit/cmake")
find_package(Houdini REQUIRED)

# Create an interface library for Houdini & OpenVDB
add_library(HoudiniVDB INTERFACE)
target_link_libraries(HoudiniVDB INTERFACE Houdini ${openvdb_lib})
target_include_directories(HoudiniVDB INTERFACE
        $ENV{HFS}/toolkit/include
)

# Add LibTorch to path
list(APPEND CMAKE_PREFIX_PATH "libtorch")

# Add CUDA kernel library
add_library(kernels STATIC Kernel.cu)
target_link_libraries(kernels PUBLIC CUDA::cudart)
set_target_properties(kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

# Add leaf extractor executable
add_executable(leafextractor
        LeafExtractor.cpp
        LeafExtractor.hpp)
target_link_libraries(leafextractor PRIVATE HoudiniVDB)

# Add decoder executable
add_executable(vqvdb_decoder VQVAE_Decoder.cpp)
target_include_directories(vqvdb_decoder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vqvdb_decoder PRIVATE
        HoudiniVDB
        ${TORCH_LIBRARIES}
        kernels
)

pybind11_add_module(
        vdb_leaf_extractor  # Module name
        cpp_extractor.cpp       # Source file
)

# Link against OpenVDB
target_link_libraries(vdb_leaf_extractor PRIVATE HoudiniVDB TBB::tbb)
