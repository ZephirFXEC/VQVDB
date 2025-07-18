cmake_minimum_required(VERSION 3.28)
project(VQVDB LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

# ─────────── Build options ─────────────────────────────────────────────────
option(ENABLE_TORCH_CPU  "Build Torch-CPU back-end"         ON)
option(ENABLE_IREE_CUDA  "Build IREE-CUDA back-end"         OFF)

# ─────────── Core include / source lists ──────────────────────────────────
set(VQVDB_CORE_SRCS
        src/VQVDB/Utils/Utils.cpp
        src/VQVDB/Utils/Profiler.cpp
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

add_compile_options(/O2 /Gy /Zc:inline)
add_link_options(/OPT:REF /OPT:ICF /INCREMENTAL:NO)

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

# Resolve dll dependencies

# Install

include(GNUInstallDirs)

set(_omp_dir "C:/Windows/System32")

# Torch CPU needs libomp.dll but it is not found by GET_RUNTIME_DEPENDENCIES
file(GLOB _omp_dlls
     LIST_DIRECTORIES FALSE
     "${_omp_dir}/libomp*.dll")

if(NOT _omp_dlls)
    message(FATAL_ERROR "No OpenMP runtime DLL (libomp*.dll) found in ${_omp_dir}")
endif()

list(GET _omp_dlls 0 LIBOMP_DLL_PATH)
message(STATUS "Found OpenMP runtime: ${LIBOMP_DLL_PATH}")

install(FILES "${LIBOMP_DLL_PATH}"
        DESTINATION ${CMAKE_INSTALL_BINDIR}) 

install(CODE [[
    file(GET_RUNTIME_DEPENDENCIES
         EXECUTABLES  $<TARGET_FILE:SOP_VQVDB_Encoder>
         RESOLVED_DEPENDENCIES_VAR RESOLVED_DEPS
         UNRESOLVED_DEPENDENCIES_VAR UNRESOLVED_DEPS
         PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-" "hvsifiletrust" "pdmutilities" "aclui" "SOP_VQVDB"
         POST_EXCLUDE_REGEXES ".*system32/.*\\.dll")

    foreach(FILE ${RESOLVED_DEPS})
        file(INSTALL
             DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
             TYPE SHARED_LIBRARY
             FOLLOW_SYMLINK_CHAIN
             FILES "${FILE}"
        )
                
    endforeach()
]])

# Check which cuda dlls are really needed, for now we copy everything
file(GLOB CUDA_DLLS
     LIST_DIRECTORIES FALSE
#      "${CUDAToolkit_BIN_DIR}/cusparse*.dll"
     "${CUDAToolkit_BIN_DIR}/*.dll"
#      "${CUDAToolkit_BIN_DIR}/cublas*.dll"
#      "${CUDAToolkit_BIN_DIR}/cudnn*.dll"
#      "${CUDAToolkit_BIN_DIR}/cufft*.dll"
#      "${CUDAToolkit_BIN_DIR}/cusolver*.dll"
     )

install(FILES ${CUDA_DLLS}
        DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES ${CMAKE_SOURCE_DIR}/houdini_package/vqvdb.json
        DESTINATION houdini_package)

install(FILES $<TARGET_FILE:SOP_VQVDB_Encoder> $<TARGET_FILE:SOP_VQVDB_Decoder>
        DESTINATION dso)