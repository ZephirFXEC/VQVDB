cmake_minimum_required(VERSION 3.28)
project(VQVDB LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

# ─────────── Build options ─────────────────────────────────────────────────
option(ENABLE_TORCH_CPU  "Build Torch-CPU back-end"         ON)
option(ENABLE_IREE_CUDA  "Build IREE-CUDA back-end"         OFF)

# ─────────── Core include / source lists ──────────────────────────────────
set(VQVDB_CORE_SRCS
        src/VQVDB/Utils/Utils.cpp
        src/VQVDB/Utils/VQVDB_Reader.cpp
        src/VQVDB/VQVAECodec.cpp
)

set(VQVDB_CORE_HDRS
        src/VQVDB/Utils/Utils.hpp
        src/VQVDB/Utils/VQVDB_Reader.hpp
        src/VQVDB/Utils/Profiler.hpp
        src/VQVDB/VQVAECodec.hpp
        src/VQVDB/Backend/IVQVAECodec.hpp
)

# ─────────── Houdini / OpenVDB (unchanged) ────────────────────────────────
list(APPEND CMAKE_PREFIX_PATH "$ENV{HFS}/toolkit/cmake")
find_package(Houdini REQUIRED)

if (UNIX)
    set(openvdb_lib "$ENV{HFS}/dsolib/libopenvdb_sesi.so")
else ()
    set(openvdb_lib "$ENV{HFS}/custom/houdini/dsolib/openvdb_sesi.lib")
endif ()

add_library(HoudiniVDB INTERFACE)
target_link_libraries(HoudiniVDB INTERFACE Houdini ${openvdb_lib})
target_include_directories(HoudiniVDB INTERFACE
        $ENV{HFS}/toolkit/include
)

# ─────────── Optional back-ends ───────────────────────────────────────────
set(BACKEND_LIBS "")

# ----- Torch-CPU ----------------------------------------------------------
if (ENABLE_TORCH_CPU)
    message(STATUS "Building Torch back-end")
    list(APPEND CMAKE_PREFIX_PATH "libtorch")
    find_package(Torch REQUIRED)

    add_library(vq_backend_torch_cpu STATIC
            src/VQVDB/Backend/TorchBackend.cpp
            src/VQVDB/Backend/TorchBackend.hpp
    )
    target_link_libraries(vq_backend_torch_cpu PUBLIC ${TORCH_LIBRARIES})

    list(APPEND BACKEND_LIBS vq_backend_torch_cpu)
endif ()

# ----- IREE-CUDA ----------------------------------------------------------
#[[if (ENABLE_IREE_CUDA)
    message(STATUS "Building IREE CUDA back-end")
    find_package(CUDAToolkit REQUIRED)
    find_package(iree-runtime REQUIRED COMPONENTS hal_cuda runtime)

    add_library(vq_backend_iree STATIC
            src/VQVDB/Backend/IreeBackend.cpp
            src/VQVDB/Backend/IreeBackend.hpp
    )
    target_link_libraries(vq_backend_iree
            PUBLIC iree::runtime
            iree::runtime::hal_cuda
    )
    list(APPEND BACKEND_LIBS vq_backend_iree)
endif ()]]

# ─────────── VQVDB umbrella static lib ────────────────────────────────────
add_library(VQVDB STATIC
        ${VQVDB_CORE_SRCS} ${VQVDB_CORE_HDRS}
)

# All chosen back-ends become PRIVATE deps of VQVDB
target_link_libraries(VQVDB
        PUBLIC HoudiniVDB
        PUBLIC ${BACKEND_LIBS}
)

target_include_directories(VQVDB PUBLIC
        "src/VQVDB"
        "src/VQVDB/Utils"
)

# ─────────── Helper to build Houdini SOPs (unchanged except linking) ─────
function(add_sop_library LIBNAME)
    add_library(${LIBNAME} SHARED ${ARGN})

    target_link_libraries(${LIBNAME} PRIVATE
            HoudiniVDB
            VQVDB
    )

    target_include_directories(${LIBNAME} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_BINARY_DIR}"
            "$ENV{HFS}/toolkit/include"
    )
    target_compile_features(${LIBNAME} PUBLIC cxx_std_17)

    if (DEFINED ENV{DSO_Path})
        houdini_configure_target(${LIBNAME} "$ENV{DSO_Path}")
    else ()
        houdini_configure_target(${LIBNAME})
    endif ()
endfunction()

# ─────────── SOP plug-ins themselves ──────────────────────────────────────
houdini_generate_proto_headers(FILES
        "src/SOP/SOP_VQVDB_Encoder.cpp"
        "src/SOP/SOP_VQVDB_Decoder.cpp"
)

add_sop_library(SOP_VQVDB_Encoder
        "src/SOP/SOP_VQVDB_Encoder.cpp"
        "src/SOP/SOP_VQVDB_Encoder.hpp"
)

add_sop_library(SOP_VQVDB_Decoder
        "src/SOP/SOP_VQVDB_Decoder.cpp"
        "src/SOP/SOP_VQVDB_Decoder.hpp"
)

# ─────────── Summary ──────────────────────────────────────────────────────
message(STATUS "Enabled back-ends: ${BACKEND_LIBS}")