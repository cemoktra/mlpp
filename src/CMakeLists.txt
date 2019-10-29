cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADER 
    model_interface.h
    regression.h
    linreg.h
    logreg.h
    traintest.h
    polyfeatures.h
    normalize.h
    kfold.h
    oneforall.h
)

set(SOURCES
    regression.cpp
    linreg.cpp
    logreg.cpp
    traintest.cpp
    polyfeatures.cpp
    normalize.cpp
    kfold.cpp
    oneforall.cpp
)

add_library(MachineLearning ${HEADER} ${SOURCES})
target_link_libraries(MachineLearning tbb eigen)