cmake_minimum_required(VERSION 3.5)

project(MachineLearning VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
add_compile_options("-msse")



configure_file(cmake/CMakeLists.txt.in download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
endif()

add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/eigen-src
                 ${CMAKE_CURRENT_BINARY_DIR}/eigen-build
                 EXCLUDE_FROM_ALL)

add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/benchmark-src
                 ${CMAKE_CURRENT_BINARY_DIR}/benchmark-build
                 EXCLUDE_FROM_ALL)          


include("${CMAKE_SOURCE_DIR}/cmake/conda.cmake")

add_subdirectory(src)
add_subdirectory(example)
add_subdirectory(test)
add_subdirectory(benchmark)