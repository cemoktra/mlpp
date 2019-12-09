cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_LINK_WHAT_YOU_USE ON)

set(HEADER 
    linreg.h
)

set(SOURCES
    linreg.cpp
)

include_directories("${PROJECT_SOURCE_DIR}/lib/xtensor-blas/include")

add_library(mlpp_regression ${HEADER} ${SOURCES})
target_link_libraries(mlpp_regression mlpp_core xtensor-blas)
target_link_libraries(mlpp_regression lapack openblas)
target_include_directories(mlpp_regression PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")