cmake_minimum_required(VERSION 3.28)
project(DWTVDB LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "$ENV{HFS}/toolkit/cmake")

find_package(Houdini REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(cxxopts CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

if (UNIX)
    set(openvdb_lib "$ENV{HFS}/dsolib/libopenvdb_sesi.so")
else ()
    set(openvdb_lib "$ENV{HFS}/custom/houdini/dsolib/openvdb_sesi.lib")
endif ()

add_library(HoudiniVDB INTERFACE)
target_link_libraries(HoudiniVDB INTERFACE Houdini ${openvdb_lib})

target_include_directories(HoudiniVDB INTERFACE "$ENV{HFS}/toolkit/include")


add_executable(DWTVDB
        main.cpp
        VDBStreamReader.cpp
        VDBStreamReader.hpp
        WaveletCompressor.cpp
        WaveletCompressor.hpp
        Logger.hpp
        IncrementalPCA.hpp
        Wavelet.hpp
)

target_link_libraries(DWTVDB
        PUBLIC
        HoudiniVDB
        Eigen3::Eigen
        cxxopts::cxxopts
        spdlog::spdlog_header_only
)