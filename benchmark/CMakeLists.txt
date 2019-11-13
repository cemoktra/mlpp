cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    matrix.cpp    
)

add_executable(mlpp_benchmark ${SOURCES})
target_link_libraries(mlpp_benchmark benchmark mlpp_regression)