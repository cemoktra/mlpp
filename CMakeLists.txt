cmake_minimum_required(VERSION 3.5)

project(MachineLearning VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(XTENSOR_USE_XSIMD 1)
#set(XTENSOR_USE_TBB 1)

include(ExternalProject)
ExternalProject_Add(external_xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
    GIT_TAG 0.6.7
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/download/xtl-src
    BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/download/xtl-bin
    INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/xtl
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/lib/xtl
    STEP_TARGETS install
)
ExternalProject_Add(external_xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
    GIT_TAG 7.4.1
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/download/xsimd-src
    BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/download/xsimd-bin
    INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/xsimd
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/lib/xsimd
    STEP_TARGETS install
)
ExternalProject_Add(external_xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
    GIT_TAG 0.20.10
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/download/xtensor-src
    BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/download/xtensor-bin
    INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/xtensor
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/lib/xtensor -DCMAKE_PREFIX_PATH=${CMAKE_CURRENT_SOURCE_DIR}/lib/xtl/lib/cmake/xtl -DCMAKE_PREFIX_PATH=${CMAKE_CURRENT_SOURCE_DIR}/lib/xsimd/lib/cmake/xsimd
    DEPENDS external_xtl-install external_xsimd-install
    STEP_TARGETS install
    UPDATE_DISCONNECTED 1
)
ExternalProject_Add(external_xtensor-blas
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas.git
    GIT_TAG 0.16.1
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/download/xtensor-blas-src
    BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/download/xtensor-blas-bin
    INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/xtensor-blas
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/lib/xtensor-blas -DCMAKE_PREFIX_PATH=${CMAKE_CURRENT_SOURCE_DIR}/lib/xtl/lib/cmake/xtl;${CMAKE_CURRENT_SOURCE_DIR}/lib/xtensor/lib/cmake/xtensor
    DEPENDS external_xtensor-install
    STEP_TARGETS install
    UPDATE_DISCONNECTED 1
)

ExternalProject_Add(external_googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.10.0
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/download/googletest-src
    BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/download/googletest-bin
    INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/googletest
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/lib/googletest
    STEP_TARGETS install
    UPDATE_DISCONNECTED 1
)

add_custom_target(extralibs DEPENDS external_xtl external_xsimd external_xtensor external_xtensor-blas external_googletest)

find_package(xsimd REQUIRED HINTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/xsimd/lib/cmake/xsimd")
find_package(xtl REQUIRED HINTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/xtl/lib/cmake/xtl")
find_package(xtensor REQUIRED HINTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/xtensor/lib/cmake/xtensor")
find_package(xtensor-blas REQUIRED HINTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/xtensor-blas/lib/cmake/xtensor-blas")
#find_package(GTest REQUIRED HINS "${CMAKE_CURRENT_SOURCE_DIR}/lib/googletest/lib/cmake/GTest")

add_definitions(-DXTENSOR_USE_TBB)

# mlpp
add_subdirectory(src)
add_subdirectory(example)
#add_subdirectory(test)