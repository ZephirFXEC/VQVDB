cmake_minimum_required(VERSION 3.21)

project(vqvdb_rendering_viewer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VQVDB_RENDERING_ENABLE_WARNINGS "Enable strict warnings" ON)

if (VQVDB_RENDERING_ENABLE_WARNINGS)
    if (MSVC)
        add_compile_options(/W4 /permissive-)
    else ()
        add_compile_options(-Wall -Wextra -Wpedantic)
    endif ()
endif ()

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

add_executable(vqvdb_rendering_viewer
        src/core/main.cpp
        src/core/app.cpp
        src/core/window.cpp
        src/core/gl_context.cpp
        src/core/input_controller.cpp
        src/core/render_loop.cpp
        src/graphics/shader.cpp
        src/graphics/mesh.cpp
        src/graphics/primitives.cpp
        src/graphics/renderer.cpp
        src/ui/ui.cpp
        src/vqvdb/vqvdb_types.cpp
        src/vqvdb/vqvdb_loader.cpp
        src/vqvdb/codebook_loader.cpp
        src/vqvdb/gpu_resources.cpp
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

install(TARGETS vqvdb_rendering_viewer RUNTIME DESTINATION bin)
