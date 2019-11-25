cmake_minimum_required(VERSION 3.5)

project(MachineLearning VERSION 0.1.0)

add_definitions(-DXTENSOR_ENABLE_XSIMD)
#add_definitions(-DXTENSOR_USE_TBB)
#add_definitions(-DXTENSOR_USE_OPENMP)

# submodules
add_subdirectory(submodules/googletest)
add_subdirectory(submodules/benchmark)
add_subdirectory(submodules/zlib)

add_subdirectory(submodules/xtl)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/submodules/xtl")

add_subdirectory(submodules/xtensor)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/submodules/xtensor")

add_subdirectory(submodules/xtensor-blas)
include_directories("${PROJECT_SOURCE_DIR}/submodules/xtensor-blas/include")

include_directories(${xtl_INCLUDE_DIRS})
include_directories(${xtensor_INCLUDE_DIRS})
include_directories(${XTENSOR_BLAS_INCLUDE_DIR})

# mlpp
add_subdirectory(src)
add_subdirectory(example)
add_subdirectory(benchmark)
# #add_subdirectory(test)