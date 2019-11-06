cmake_minimum_required(VERSION 3.5)

project(MachineLearning VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(XTENSOR_USE_XSIMD 1)
#set(XTENSOR_USE_TBB 1)


# submodule
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/build/xtl/")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/build/xtensor/")
add_subdirectory(googletest)
add_subdirectory(xtl)
add_subdirectory(xtensor)
add_subdirectory(xtensor-blas)

# mlpp
add_subdirectory(src)
#add_subdirectory(example)
#add_subdirectory(test)