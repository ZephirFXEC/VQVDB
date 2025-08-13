cmake_minimum_required(VERSION 3.28)
project(VQVDB LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─────────── Build options ─────────────────────────────────────────────────
option(ENABLE_TORCH_BACKEND "Build the LibTorch backend" OFF)
option(ENABLE_ONNX_BACKEND "Build the ONNXRuntime backend" ON)

# ─────────── Houdini / OpenVDB ─────────────────────────────────────────────
list(APPEND CMAKE_PREFIX_PATH "$ENV{HFS}/toolkit/cmake")
find_package(Houdini REQUIRED)

if (UNIX)
    set(openvdb_lib "$ENV{HFS}/dsolib/libopenvdb_sesi.so")
else ()
    set(openvdb_lib "$ENV{HFS}/custom/houdini/dsolib/openvdb_sesi.lib")
    set(onnxruntime_lib "$ENV{HFS}/custom/houdini/dsolib/onnxruntime.lib")
endif ()

add_library(HoudiniVDB INTERFACE)
target_link_libraries(HoudiniVDB INTERFACE Houdini ${openvdb_lib})

target_include_directories(HoudiniVDB INTERFACE "$ENV{HFS}/toolkit/include")

add_compile_options(${_houdini_compile_options})

# ─────────── 1. CORE INTERFACE (Header-only) ───────────────────────────────
# Defines the IVQVAECodec interface. No .cpp files, no ML dependencies.
# ---------------------------------------------------------------------------
add_library(vqvdb_core INTERFACE)
target_include_directories(vqvdb_core INTERFACE "src")

# ─────────── 2. ORCHESTRATOR LIBRARY ───────────────────────────────────────
# The high-level VQVAECodec logic. Depends on the core interface, not on any
# specific backend implementation.
# ---------------------------------------------------------------------------
add_library(vqvdb_orchestrator STATIC
        src/orchestrator/VQVAECodec.cpp
        src/orchestrator/VQVAECodec.hpp
        src/utils/VQVDB_Reader.cpp
        src/utils/VQVDB_Reader.hpp
        src/utils/Utils.cpp
        src/utils/Utils.hpp
        src/utils/Profiler.hpp
)
target_include_directories(vqvdb_orchestrator PUBLIC "src")
target_link_libraries(vqvdb_orchestrator
        PUBLIC
        vqvdb_core
        HoudiniVDB
)

# ─────────── 3. BACKEND IMPLEMENTATIONS ────────────────────────────────────
# Each backend is a separate static library that links to its specific ML framework.
# They all depend on the `vqvdb_core` to get the interface definition.
# ---------------------------------------------------------------------------
set(ENABLED_BACKEND_TARGETS "")
set(ENABLED_BACKEND_DEFINITIONS "")

# ----- LibTorch Backend ----------------------------------------------------
if (ENABLE_TORCH_BACKEND)
    message(STATUS "Building Torch backend")
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
    set(NVTX3_INCLUDE_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include/nvtx3")
    include_directories(${NVTX3_INCLUDE_DIR})

    LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

    find_package(Torch REQUIRED)

    if (NOT TARGET CUDA::nvToolsExt)
        add_library(nvtx3_dummy INTERFACE)
        target_include_directories(nvtx3_dummy INTERFACE "${NVTX3_INCLUDE_DIR}")
        add_library(CUDA::nvToolsExt ALIAS nvtx3_dummy)
    endif ()

    add_library(vqvdb_backend_torch STATIC
            src/backends/torch/TorchBackend.cpp
            src/backends/torch/TorchBackend.hpp
    )
    target_link_libraries(vqvdb_backend_torch
            PUBLIC
            vqvdb_core
            ${TORCH_LIBRARIES}
    )

    list(APPEND ENABLED_BACKEND_TARGETS vqvdb_backend_torch)
    list(APPEND ENABLED_BACKEND_DEFINITIONS "ENABLE_TORCH_BACKEND")
endif ()

# ----- ONNX Backend --------------------------------------------------------
if (ENABLE_ONNX_BACKEND)
    message(STATUS "Building ONNX backend (using Houdini's runtime)")
    find_package(CUDAToolkit REQUIRED)

    add_library(vqvdb_backend_onnx STATIC
            src/backends/onnx/OnnxBackendFactory.cpp
            src/backends/onnx/OnnxBackendFactory.hpp
            src/backends/onnx/OnnxBackend_CPU.cpp
            src/backends/onnx/OnnxBackend_CPU.hpp
            src/backends/onnx/OnnxBackend_Cuda.cpp
            src/backends/onnx/OnnxBackend_Cuda.hpp
    )
    # Link against HoudiniV. This provides the include paths for onnxruntime
    # headers AND links the necessary onnxruntime library.
    target_link_libraries(vqvdb_backend_onnx
            PUBLIC vqvdb_core Houdini CUDA::cudart ${onnxruntime_lib}
    )

    list(APPEND ENABLED_BACKEND_TARGETS vqvdb_backend_onnx)
    list(APPEND ENABLED_BACKEND_DEFINITIONS "ENABLE_ONNX_BACKEND")
endif ()

# ─────────── 4. THE FACTORY ────────────────────────────────────────────────
# This is the bridge. It contains ONLY the IVQVAECodec::create implementation.
# It links to all enabled backends so it can instantiate them.
# ---------------------------------------------------------------------------
add_library(vqvdb_factory STATIC src/core/IVQVAECodec.cpp)
target_link_libraries(vqvdb_factory
        PUBLIC
        vqvdb_core
        ${ENABLED_BACKEND_TARGETS} # Link to the actual backend libs
)
# Pass definitions to IVQVAECodec.cpp so it knows which headers to include
target_compile_definitions(vqvdb_factory PRIVATE ${ENABLED_BACKEND_DEFINITIONS})


# ─────────── 5. HOUDINI SOPs ───────────────────────
function(add_sop_library LIBNAME)
    add_library(${LIBNAME} SHARED ${ARGN})

    # The SOP links to the ORCHESTRATOR and the FACTORY.
    # The factory transitively pulls in all the enabled backend implementations.
    target_link_libraries(${LIBNAME} PRIVATE
            vqvdb_orchestrator
            vqvdb_factory
    )

    # Standard Houdini configuration
    target_include_directories(${LIBNAME} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_BINARY_DIR}"
    )
    if (DEFINED ENV{DSO_Path})
        houdini_configure_target(${LIBNAME} "$ENV{DSO_Path}")
    else ()
        houdini_configure_target(${LIBNAME})
    endif ()
endfunction()

# ─────────── SOP Plug-in Definitions ──────────────────────────────────────
houdini_generate_proto_headers(FILES
        "src/SOP/SOP_VQVDB_Encoder.cpp"
        "src/SOP/SOP_VQVDB_Decoder.cpp"
)

# Create a SINGLE shared library for all VQ-VDB SOPs
add_sop_library(SOP_VQVDB
        # The new registration file
        "src/SOP/SOP_Register.cpp"

        # The Encoder files
        "src/SOP/SOP_VQVDB_Encoder.cpp"
        "src/SOP/SOP_VQVDB_Encoder.hpp"

        # The Decoder files
        "src/SOP/SOP_VQVDB_Decoder.cpp"
        "src/SOP/SOP_VQVDB_Decoder.hpp"
)

# ─────────── Summary ──────────────────────────────────────────────────────
message(STATUS "Enabled backend targets: ${ENABLED_BACKEND_TARGETS}")
# Resolve dll dependencies

# Install

include(GNUInstallDirs)

set(_omp_dir "C:/Windows/System32")

# Torch CPU needs libomp.dll but it is not found by GET_RUNTIME_DEPENDENCIES
file(GLOB _omp_dlls
        LIST_DIRECTORIES FALSE
        "${_omp_dir}/libomp*.dll")

if (NOT _omp_dlls)
    message(FATAL_ERROR "No OpenMP runtime DLL (libomp*.dll) found in ${_omp_dir}")
endif ()

list(GET _omp_dlls 0 LIBOMP_DLL_PATH)
message(STATUS "Found OpenMP runtime: ${LIBOMP_DLL_PATH}")

install(FILES "${LIBOMP_DLL_PATH}"
        DESTINATION ${CMAKE_INSTALL_BINDIR})

install(CODE [[
    file(GET_RUNTIME_DEPENDENCIES
         EXECUTABLES  $<TARGET_FILE:SOP_VQVDB>
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

file(GLOB CUDA_DLLS
        LIST_DIRECTORIES FALSE
        "${CUDAToolkit_BIN_DIR}/cublas64_12.dll"
        "${CUDAToolkit_BIN_DIR}/cublasLt64_12.dll"
        "${CUDAToolkit_BIN_DIR}/cufft64_11.dll"
        "C:/Program Files/NVIDIA/CUDNN/v8.9.2/bin/cudnn64_8.dll" # Houdini 20.5 uses cuDNN 8 :')
)

install(FILES ${CUDA_DLLS}
        DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES ${CMAKE_SOURCE_DIR}/houdini_package/vqvdb.json
        DESTINATION houdini_package)

install(FILES $<TARGET_FILE:SOP_VQVDB> $<TARGET _FILE:SOP_VQVDB_Decoder>
        DESTINATION dso)