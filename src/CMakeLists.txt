cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADER 
    regression.h
    linreg.h
    traintest.h
    polyfeatures.h
    normalize.h
    kfold.h
)

set(SOURCES
    regression.cpp
    linreg.cpp
    traintest.cpp
    polyfeatures.cpp
    normalize.cpp
    kfold.cpp
)

add_library(MachineLearning ${HEADER} ${SOURCES})
target_link_libraries(MachineLearning tbb eigen)