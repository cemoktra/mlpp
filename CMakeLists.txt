cmake_minimum_required(VERSION 3.5)

project(MachineLearning VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

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


unset(CONDA_BIN_PATH)
unset(CONDA_BIN_PATH CACHE)

if(WIN32)
    set(CONDA_BIN_NAME "conda.bat")
else()
    set(CONDA_BIN_NAME "conda")
endif()

find_program(CONDA_BIN_PATH name "${CONDA_BIN_NAME}" PATHS "/usr/bin/")
if (NOT CONDA_BIN_PATH) 
    find_program(CONDA_BIN_PATH name "${CONDA_BIN_NAME}" PATHS "$ENV{PROGRAMDATA}/Anaconda3/condabin/")
    if (NOT CONDA_BIN_PATH) 
        find_program(CONDA_BIN_PATH name "${CONDA_BIN_NAME}" PATHS "$ENV{PROGRAMDATA}/Miniconda3/condabin/")
    endif()
endif()
if (NOT CONDA_BIN_PATH)
    message(FATAL_ERROR "${CONDA_BIN_NAME} not found! conda is required to setup libraries")
endif()

message(STATUS "upgrading conda packages")
execute_process(COMMAND "${CONDA_BIN_PATH}" "env" "update" "--prefix" "${CMAKE_SOURCE_DIR}/conda" "-f" "${CMAKE_SOURCE_DIR}/environment.yml" "--prune" OUTPUT_QUIET RESULT_VARIABLE CONDA_RC)

if (CONDA_RC EQUAL 0) 
    message(STATUS "conda upgrade done")
else()
    message(FATAL_ERROR "conda upgrade failed with return code ${CONDA_RC}")
endif()

if(WIN32)
    set(CONDA_LIB_BIN "${CMAKE_SOURCE_DIR}/conda/Library/bin")
    include_directories("${CMAKE_SOURCE_DIR}/conda/Library/include")
    link_directories("${CMAKE_SOURCE_DIR}/conda/Library/lib")
else()
    include_directories("${CMAKE_SOURCE_DIR}/conda/include")
    link_directories("${CMAKE_SOURCE_DIR}/conda/lib")
endif()
                 

add_subdirectory(src)
add_subdirectory(example)
add_subdirectory(test)
add_subdirectory(benchmark)