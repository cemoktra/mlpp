cmake_minimum_required(VERSION 3.5)

project(MachineLearning VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_program(CONDA name conda HINTS "/usr/bin/;$ENV{PROGRAMDATA}/Anaconda3/condabin/conda.bat;$ENV{PROGRAMDATA}/Miniconda3/condabin/conda.bat")
if (NOT CONDA)
    message(SEND_ERROR "conda not found! conda is r equired to setup libraries")
endif()
message(STATUS "upgrading conda packages")
execute_process(COMMAND "${CONDA}" "env" "update" "--prefix" "${CMAKE_SOURCE_DIR}/conda" "-f" "${CMAKE_SOURCE_DIR}/environment.yml" "--prune" OUTPUT_QUIET RESULT_VARIABLE CONDA_RC)

if (CONDA_RC EQUAL 0) 
    message(STATUS "conda upgrade done")
else()
    message(SEND_ERROR "conda upgrade failed with return code ${CONDA_RC}")
endif()

if(WIN32)
    include_directories("${CMAKE_SOURCE_DIR}/conda/Library/include")
    link_directories("${CMAKE_SOURCE_DIR}/conda/Library/lib")
else()
    include_directories("${CMAKE_SOURCE_DIR}/conda/include")
    link_directories("${CMAKE_SOURCE_DIR}/conda/lib")
endif()

set(XTENSOR_USE_XSIMD 1)
set(XTENSOR_USE_TBB 1)

# mlpp
add_subdirectory(src)
add_subdirectory(example)
# #add_subdirectory(test)