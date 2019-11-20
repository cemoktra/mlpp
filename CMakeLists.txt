cmake_minimum_required(VERSION 3.5)

project(MachineLearning VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    # TODO find conda
    set (CONDA "$ENV{PROGRAMDATA}/Anaconda3/condabin/conda.bat")
    # set (CONDA "$ENV{PROGRAMDATA}/Miniconda3/condabin/conda.bat")
else()
    set (CONDA "conda")
endif()
execute_process(COMMAND "${CONDA}" "env" "update" "--prefix" "${CMAKE_SOURCE_DIR}/conda" "-f" "${CMAKE_SOURCE_DIR}/environment.yml" "--prune" OUTPUT_QUIET)

include_directories("${CMAKE_SOURCE_DIR}/conda/Library/include")
link_directories("${CMAKE_SOURCE_DIR}/conda/Library/lib")

set(XTENSOR_USE_XSIMD 1)
set(XTENSOR_USE_TBB 1)

# mlpp
add_subdirectory(src)
add_subdirectory(example)
# #add_subdirectory(test)