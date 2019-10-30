cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADER 
    model_interface.h
    regression.h
    linreg.h
    logreg.h
    multinomial_logreg.h
    traintest.h
    polyfeatures.h
    normalize.h
    kfold.h
    oneforone.h
    one_hot.h
)

set(SOURCES
    regression.cpp
    linreg.cpp
    logreg.cpp
    multinomial_logreg.cpp
    traintest.cpp
    polyfeatures.cpp
    normalize.cpp
    kfold.cpp
    oneforone.cpp
    one_hot.cpp
)

add_library(MachineLearning ${HEADER} ${SOURCES})
target_link_libraries(MachineLearning tbb eigen)