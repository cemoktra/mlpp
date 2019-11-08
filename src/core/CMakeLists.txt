cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADER 
    traintest.h
    polyfeatures.h
    normalize.h
    kfold.h
    one_hot.h
    csv_reader.h
    csv_data.h
    parameters.h
    model.h
    validation_curve.h
    learning_curve.h
    vocabulary.h
    pseudo_inverse.h
    matrix.h
)

set(SOURCES
    traintest.cpp
    polyfeatures.cpp
    normalize.cpp
    kfold.cpp
    one_hot.cpp
    csv_reader.cpp
    csv_data.cpp
    parameters.cpp
    validation_curve.cpp
    learning_curve.cpp
    vocabulary.cpp
    pseudo_inverse.cpp
    matrix.cpp
)

add_definitions("-mavx")

add_library(mlpp_core ${HEADER} ${SOURCES})
target_link_libraries(mlpp_core tbb eigen)
target_include_directories(mlpp_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")