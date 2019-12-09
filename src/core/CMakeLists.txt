cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADER 
    traintest.h
    kfold.h
    csv_reader.h
    csv_data.h
    parameters.h
    model.h
    #validation_curve.h
    #learning_curve.h
    mnist_data.h
    scoped_timer.h
)

set(SOURCES
    traintest.cpp
    kfold.cpp
    csv_reader.cpp
    csv_data.cpp
    parameters.cpp
    #validation_curve.cpp
    #learning_curve.cpp
    mnist_data.cpp
)

add_library(mlpp_core ${HEADER} ${SOURCES})
target_link_libraries(mlpp_core tbb zlib xtensor)
target_include_directories(mlpp_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")