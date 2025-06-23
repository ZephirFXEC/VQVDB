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

add_executable(VQVAE_v2
        VQVDB_CPP/VQVAE_v2.cpp
        VQVDB_CPP/VQVAE_v2.hpp
        VQVDB_CPP/VQVAE_Training.cpp)
target_link_libraries(VQVAE_v2 PRIVATE
        HoudiniVDB
        ${TORCH_LIBRARIES}
)

# Add decoder executable
add_executable(vqvdb_decoder VQVDB_CPP/VQVAE_Decoder.cpp VQVDB_CPP/VQVAE_Decoder.hpp
        VQVDB_CPP/Utils/VQVDB_Reader.cpp
        VQVDB_CPP/Utils/VQVDB_Reader.hpp)

target_include_directories(vqvdb_decoder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vqvdb_decoder PRIVATE
        HoudiniVDB
        ${TORCH_LIBRARIES}
)

add_executable(vqvdb_encoder VQVDB_CPP/VQVAE_Encoder.cpp VQVDB_CPP/VQVAE_Encoder.hpp)

target_include_directories(vqvdb_encoder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vqvdb_encoder PRIVATE
        HoudiniVDB
        ${TORCH_LIBRARIES}
)


# VQVDB_CPP interface library
add_library(VQVDB_CPP STATIC
        VQVDB_CPP/VQVAE_Decoder.hpp VQVDB_CPP/VQVAE_Encoder.hpp VQVDB_CPP/VQVAE_Encoder.cpp VQVDB_CPP/VQVAE_Decoder.cpp
        VQVDB_CPP/Utils/Utils.cpp
        VQVDB_CPP/Utils/Utils.hpp)
target_include_directories(VQVDB_CPP PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/VQVDB_CPP"
        "${CMAKE_CURRENT_SOURCE_DIR}/VQVDB_CPP/Utils"
)

target_link_libraries(VQVDB_CPP PUBLIC
        HoudiniVDB
        ${TORCH_LIBRARIES}
)

# ------- Helper function for SOP libraries -------
function(add_sop_library LIBNAME SRC_FILES)
    add_library(${LIBNAME} SHARED ${SRC_FILES})

    target_link_libraries(${LIBNAME} PRIVATE
            HoudiniVDB
            VQVDB_CPP
            ${TORCH_LIBRARIES}
    )

    target_include_directories(${LIBNAME} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_BINARY_DIR}"
            "$ENV{HFS}/toolkit/include"
    )

    target_compile_features(${LIBNAME} PUBLIC cxx_std_17)

    # Configure for Houdini
    if (DEFINED ENV{DSO_Path})
        houdini_configure_target(${LIBNAME} "$ENV{DSO_Path}")
    else ()
        houdini_configure_target(${LIBNAME})
    endif ()
endfunction()

# ------- Generate Houdini protocol headers -------
houdini_generate_proto_headers(FILES
        "SOP/SOP_VQVDB_Encoder.cpp"
        "SOP/SOP_VQVDB_Decoder.cpp"
)

# ------- Create SOP libraries -------
add_sop_library(SOP_VQVDB_Encoder
        "SOP/SOP_VQVDB_Encoder.cpp"
        "SOP/SOP_VQVDB_Encoder.hpp"
)

#[[
add_sop_library(SOP_VQVDB_Decoder
        "SOP/SOP_VQVDB_Decoder.cpp"
        "SOP/SOP_VQVDB_Decoder.hpp"
)]]
