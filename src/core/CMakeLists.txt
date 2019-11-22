cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADER 
    traintest.h
    polyfeatures.h
    normalize.h
    standard_scale.h
    kfold.h
    one_hot.h
    csv_reader.h
    csv_data.h
    parameters.h
    model.h
    #validation_curve.h
    #learning_curve.h
    vocabulary.h
    mnist_data.h
)

set(SOURCES
    traintest.cpp
    polyfeatures.cpp
    normalize.cpp
    standard_scale.cpp
    kfold.cpp
    one_hot.cpp
    csv_reader.cpp
    csv_data.cpp
    parameters.cpp
    #validation_curve.cpp
    #learning_curve.cpp
    vocabulary.cpp
    mnist_data.cpp
)

include_directories("${PROJECT_SOURCE_DIR}/lib/xtensor-blas/include")

add_library(mlpp_core ${HEADER} ${SOURCES})
target_link_libraries(mlpp_core tbb zlib)
target_include_directories(mlpp_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")