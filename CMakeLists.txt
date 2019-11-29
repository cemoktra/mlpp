cmake_minimum_required(VERSION 3.5)

project(MachineLearning VERSION 0.1.0)

add_definitions(-DXTENSOR_ENABLE_XSIMD)
#add_definitions(-DXTENSOR_USE_TBB)
#add_definitions(-DXTENSOR_USE_OPENMP)

# submodules
add_subdirectory(submodules/googletest EXCLUDE_FROM_ALL )
add_subdirectory(submodules/benchmark EXCLUDE_FROM_ALL )
add_subdirectory(submodules/zlib EXCLUDE_FROM_ALL )

add_subdirectory(submodules/xtl EXCLUDE_FROM_ALL )
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/submodules/xtl")
add_subdirectory(submodules/xtensor EXCLUDE_FROM_ALL )
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/submodules/xtensor")
add_subdirectory(submodules/xtensor-blas EXCLUDE_FROM_ALL )

include_directories(${CMAKE_BINARY_DIR}/submodules/zlib)
include_directories(${PROJECT_SOURCE_DIR}/submodules/zlib)
include_directories(${xtl_INCLUDE_DIRS})
include_directories(${xtensor_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/submodules/xtensor-blas/include)

# mlpp
add_subdirectory(src)
add_subdirectory(example)
add_subdirectory(benchmark)
# #add_subdirectory(test)