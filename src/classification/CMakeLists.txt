cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADER
    classifier.h 
    logreg.h
    multinomial_logreg.h
    oneforone.h
    decision_tree.h
    decision_tree_node.h
    knn.h
)

set(SOURCES
    logreg.cpp
    multinomial_logreg.cpp
    oneforone.cpp
    decision_tree.cpp
    decision_tree_node.cpp
    knn.cpp
)

add_library(mlpp_classification ${HEADER} ${SOURCES})
target_link_libraries(mlpp_classification mlpp_core)
target_include_directories(mlpp_classification PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")