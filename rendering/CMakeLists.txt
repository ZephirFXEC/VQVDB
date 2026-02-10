cmake_minimum_required(VERSION 3.21)

project(vqvdb_rendering_viewer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VQVDB_RENDERING_ENABLE_WARNINGS "Enable strict warnings" ON)
option(VQVDB_RENDERING_ENABLE_TRT "Enable TensorRT decoder backend (CUDA + TensorRT required)" ON)
option(VQVDB_RENDERING_BUILD_TESTS "Build rendering unit tests" ON)
option(VQVDB_RENDERING_BUILD_VIEWER "Build rendering viewer app" ON)

if (VQVDB_RENDERING_ENABLE_WARNINGS)
    if (MSVC)
        add_compile_options(/W4 /permissive-)
    else ()
        add_compile_options(-Wall -Wextra -Wpedantic)
    endif ()
endif ()

find_package(glm CONFIG REQUIRED)
if (VQVDB_RENDERING_BUILD_VIEWER)
    find_package(OpenGL REQUIRED)
    find_package(glfw3 CONFIG REQUIRED)
    find_package(glad CONFIG REQUIRED)
    find_package(imgui CONFIG REQUIRED)
endif ()

if (VQVDB_RENDERING_ENABLE_TRT)
    find_package(CUDAToolkit REQUIRED)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            HINTS ENV TensorRT_ROOT
            PATH_SUFFIXES include)
    find_library(TENSORRT_NVINFER_LIB nvinfer
            HINTS ENV TensorRT_ROOT
            PATH_SUFFIXES lib)
    find_library(TENSORRT_NVONNXPARSER_LIB nvonnxparser
            HINTS ENV TensorRT_ROOT
            PATH_SUFFIXES lib)
    if (NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIB OR NOT TENSORRT_NVONNXPARSER_LIB)
        message(FATAL_ERROR "TensorRT not found. Set TensorRT_ROOT or disable VQVDB_RENDERING_ENABLE_TRT.")
    endif ()

    # Find TensorRT DLLs for runtime
    if (WIN32)
        find_file(TENSORRT_NVINFER_DLL nvinfer.dll
                HINTS ENV TensorRT_ROOT
                PATH_SUFFIXES bin lib)
        find_file(TENSORRT_NVONNXPARSER_DLL nvonnxparser.dll
                HINTS ENV TensorRT_ROOT
                PATH_SUFFIXES bin lib)
        if (NOT TENSORRT_NVINFER_DLL OR NOT TENSORRT_NVONNXPARSER_DLL)
            message(WARNING "TensorRT DLLs not found. Runtime may fail if DLLs are not in PATH.")
        endif ()
    endif ()
endif ()

if (VQVDB_RENDERING_BUILD_VIEWER)
    add_executable(vqvdb_rendering_viewer
            src/core/main.cpp
            src/core/app.cpp
            src/core/window.cpp
            src/core/gl_context.cpp
            src/core/input_controller.cpp
            src/core/profiler.cpp
            src/core/render_loop.cpp
            src/core/depth_pyramid.cpp
            src/graphics/shader.cpp
            src/graphics/mesh.cpp
            src/graphics/primitives.cpp
            src/graphics/renderer.cpp
            src/ui/ui.cpp
            src/ui/profiler_ui.cpp
            src/vqvdb/vqvdb_types.cpp
            src/vqvdb/vqvdb_loader.cpp
            src/vqvdb/gpu_resources.cpp
            src/vqvdb/decoder_backend.cpp
            src/vqvdb/cuda_gl_interop.cpp
            src/vqvdb/brick_cache.cpp
            src/vqvdb/scheduler.cpp
    )

    target_link_libraries(vqvdb_rendering_viewer
            PRIVATE
            OpenGL::GL
            glfw
            glad::glad
            glm::glm
            imgui::imgui
            $<$<PLATFORM_ID:Windows>:comdlg32>
    )

    if (VQVDB_RENDERING_ENABLE_TRT)
        target_compile_definitions(vqvdb_rendering_viewer PRIVATE VQVDB_RENDERING_ENABLE_TRT)
        target_include_directories(vqvdb_rendering_viewer PRIVATE ${TENSORRT_INCLUDE_DIR})
        target_link_libraries(vqvdb_rendering_viewer PRIVATE
                CUDA::cudart
                ${TENSORRT_NVINFER_LIB}
                ${TENSORRT_NVONNXPARSER_LIB}
        )

        # Copy TensorRT DLLs to output directory on Windows
        if (WIN32 AND TENSORRT_NVINFER_DLL AND TENSORRT_NVONNXPARSER_DLL)
            add_custom_command(TARGET vqvdb_rendering_viewer POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${TENSORRT_NVINFER_DLL}"
                    "$<TARGET_FILE_DIR:vqvdb_rendering_viewer>"
                    COMMENT "Copying nvinfer.dll to output directory"
            )
            add_custom_command(TARGET vqvdb_rendering_viewer POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${TENSORRT_NVONNXPARSER_DLL}"
                    "$<TARGET_FILE_DIR:vqvdb_rendering_viewer>"
                    COMMENT "Copying nvonnxparser.dll to output directory"
            )
        endif ()
    endif ()

    target_include_directories(vqvdb_rendering_viewer
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    if (WIN32)
        target_compile_definitions(vqvdb_rendering_viewer PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    endif ()

    set_target_properties(vqvdb_rendering_viewer PROPERTIES
            OUTPUT_NAME "vqvdb_rendering_viewer"
    )
endif ()

if (VQVDB_RENDERING_BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
            doctest
            GIT_REPOSITORY https://github.com/doctest/doctest.git
            GIT_TAG v2.4.12
    )
    FetchContent_MakeAvailable(doctest)
    include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)

    add_executable(vqvdb_rendering_decoder_unit_tests
            tests/decoder_backend_unit_tests.cpp
            src/vqvdb/vqvdb_types.cpp
            src/vqvdb/decoder_backend.cpp
            src/vqvdb/cuda_gl_interop.cpp
    )

    target_include_directories(vqvdb_rendering_decoder_unit_tests
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(vqvdb_rendering_decoder_unit_tests
            PRIVATE
            glm::glm
            glad::glad
            doctest::doctest
    )

    if (VQVDB_RENDERING_ENABLE_TRT)
        target_compile_definitions(vqvdb_rendering_decoder_unit_tests PRIVATE VQVDB_RENDERING_ENABLE_TRT)
        target_include_directories(vqvdb_rendering_decoder_unit_tests PRIVATE ${TENSORRT_INCLUDE_DIR})
        target_link_libraries(vqvdb_rendering_decoder_unit_tests PRIVATE
                CUDA::cudart
                ${TENSORRT_NVINFER_LIB}
                ${TENSORRT_NVONNXPARSER_LIB}
        )
        if (TARGET OpenGL::GL AND TARGET glfw)
            target_compile_definitions(vqvdb_rendering_decoder_unit_tests PRIVATE VQVDB_DECODER_TEST_GL_CONTEXT)
            target_link_libraries(vqvdb_rendering_decoder_unit_tests PRIVATE
                    OpenGL::GL
                    glfw
            )
        endif ()

        # Copy TensorRT DLLs to output directory on Windows
        if (WIN32 AND TENSORRT_NVINFER_DLL AND TENSORRT_NVONNXPARSER_DLL)
            add_custom_command(TARGET vqvdb_rendering_decoder_unit_tests POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${TENSORRT_NVINFER_DLL}"
                    "$<TARGET_FILE_DIR:vqvdb_rendering_decoder_unit_tests>"
                    COMMENT "Copying nvinfer.dll to test output directory"
            )
            add_custom_command(TARGET vqvdb_rendering_decoder_unit_tests POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${TENSORRT_NVONNXPARSER_DLL}"
                    "$<TARGET_FILE_DIR:vqvdb_rendering_decoder_unit_tests>"
                    COMMENT "Copying nvonnxparser.dll to test output directory"
            )
        endif ()
    endif ()

    doctest_discover_tests(vqvdb_rendering_decoder_unit_tests)

    add_executable(vqvdb_rendering_brick_cache_unit_tests
            tests/brick_cache_unit_tests.cpp
            src/vqvdb/vqvdb_types.cpp
            src/vqvdb/brick_cache.cpp
    )

    target_include_directories(vqvdb_rendering_brick_cache_unit_tests
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(vqvdb_rendering_brick_cache_unit_tests
            PRIVATE
            glm::glm
            glad::glad
            doctest::doctest
    )

    doctest_discover_tests(vqvdb_rendering_brick_cache_unit_tests)
endif ()

if (VQVDB_RENDERING_BUILD_VIEWER)
    install(TARGETS vqvdb_rendering_viewer RUNTIME DESTINATION bin)
endif ()
