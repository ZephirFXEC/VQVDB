cmake_minimum_required(VERSION 3.28)
project(VQVDB LANGUAGES CUDA CXX)

set(USE_CUDNN 1)
# ------- Find CUDA -----------
find_package(CUDAToolkit REQUIRED)
find_package(Torch CONFIG REQUIRED)

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
add_library(kernels STATIC VQVDB_CPP/Utils/Kernel.cu)
target_link_libraries(kernels PUBLIC CUDA::cudart)
set_target_properties(kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

# Add leaf extractor executable
add_executable(leafextractor
        VQVDB_CPP/Utils/LeafExtractor.cpp
        VQVDB_CPP/Utils/LeafExtractor.hpp)
target_link_libraries(leafextractor PRIVATE HoudiniVDB)

add_executable(VQVAE_v2
        VQVDB_CPP/VQVAE_v2.cpp
        VQVDB_CPP/VQVAE_v2.hpp
        VQVDB_CPP/VQVAE_Training.cpp)
target_link_libraries(VQVAE_v2 PRIVATE
        HoudiniVDB
        ${TORCH_LIBRARIES}
        kernels
)

# Add decoder executable
add_executable(vqvdb_decoder VQVDB_CPP/VQVAE_Decoder.cpp
        VQVDB_CPP/Utils/VQVDB_Reader.cpp
        VQVDB_CPP/Utils/VQVDB_Reader.hpp)
target_include_directories(vqvdb_decoder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vqvdb_decoder PRIVATE
        HoudiniVDB
        ${TORCH_LIBRARIES}
        kernels
)