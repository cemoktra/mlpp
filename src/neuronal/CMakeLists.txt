cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_LINK_WHAT_YOU_USE ON)

set(HEADER 
    activation.h
    backprop_layer.h
    dense_layer.h
    net.h
    solver.h
    loss.h
)

set(SOURCES
    backprop_layer.cpp
    dense_layer.cpp
    net.cpp
    solver.cpp
)

add_library(mlpp_neuronal ${HEADER} ${SOURCES})
target_link_libraries(mlpp_neuronal xtensor xtensor-blas)
target_link_libraries(mlpp_neuronal lapack openblas)
target_include_directories(mlpp_neuronal PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")