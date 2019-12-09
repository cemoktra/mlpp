cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_LINK_WHAT_YOU_USE ON)

set(HEADER 
    kmeans.h
)

set(SOURCES
    kmeans.cpp
)

add_library(mlpp_clustering ${HEADER} ${SOURCES})
target_link_libraries(mlpp_clustering mlpp_core xtensor-blas)
target_link_libraries(mlpp_clustering lapack openblas)
target_include_directories(mlpp_clustering PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")