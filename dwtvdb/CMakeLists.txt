cmake_minimum_required(VERSION 3.28)
project(DWTVDB LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "$ENV{HFS}/toolkit/cmake")

find_package(Houdini REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_path(POCKETFFT_INCLUDE_DIRS "pocketfft_hdronly.h")
find_package(cxxopts CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

if (UNIX)
    set(ZLIB_LIBRARY "$ENV{HFS}/dsolib/zdll.so")
    set(openvdb_lib "$ENV{HFS}/dsolib/libopenvdb_sesi.so")
else ()
    set(ZLIB_LIBRARY "$ENV{HFS}/custom/houdini/dsolib/zdll.lib")
    set(openvdb_lib "$ENV{HFS}/custom/houdini/dsolib/openvdb_sesi.lib")
endif ()

if (NOT ZLIB_LIBRARY)
    message(FATAL_ERROR "Could not find Houdini's zlib header or library.")
else ()
    message(STATUS "Found ZLIB library: ${ZLIB_LIBRARY}")
endif ()

add_library(HoudiniVDB INTERFACE)
target_link_libraries(HoudiniVDB INTERFACE Houdini ${openvdb_lib})

target_include_directories(HoudiniVDB INTERFACE "$ENV{HFS}/toolkit/include")


add_executable(DWTVDB
        main.cpp
        VDBStreamReader.cpp
        VDBStreamReader.hpp
        Logger.hpp
        LeafActivityMask.hpp
        Wavelet.hpp
        Compressor.hpp
        Compressor.cpp
)

target_include_directories(DWTVDB PRIVATE ${POCKETFFT_INCLUDE_DIRS})

target_link_libraries(DWTVDB
        PUBLIC
        HoudiniVDB
        Eigen3::Eigen
        cxxopts::cxxopts
        spdlog::spdlog_header_only
        ${ZLIB_LIBRARY}
)