cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options("-march=native")

add_subdirectory(core)
add_subdirectory(regression)
add_subdirectory(classification)