cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADER 
    linreg.h
    traintest.h
    polyfeatures.h
    kfold.h
)

set(SOURCES
    linreg.cpp
    traintest.cpp
    polyfeatures.cpp
    kfold.cpp
)

add_library(LinearRegression ${HEADER} ${SOURCES})
target_link_libraries(LinearRegression tbb)