cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADER 
    polyfeatures.h
    one_hot.h
    scaler.h
    vocabulary.h
    pca.h
)

set(SOURCES
    polyfeatures.cpp
    one_hot.cpp
    scaler.cpp
    vocabulary.cpp
    pca.cpp
)

add_library(mlpp_preprocessing ${HEADER} ${SOURCES})
target_link_libraries(mlpp_preprocessing mlpp_core tbb xtensor xtensor-blas)
target_include_directories(mlpp_preprocessing PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")