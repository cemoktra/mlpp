cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADER 
    logreg.h
    multinomial_logreg.h
    oneforone.h
)

set(SOURCES
    logreg.cpp
    multinomial_logreg.cpp
    oneforone.cpp
)

add_library(mlpp_classification ${HEADER} ${SOURCES})
target_link_libraries(mlpp_classification mlpp_core)
target_include_directories(mlpp_classification PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")