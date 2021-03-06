cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADER
    classifier.h 
    logreg.h
    multinomial_logreg.h
    oneforone.h
    knn.h
    decision_tree.h
    decision_tree_node.h
    random_forest.h
    naive_bayes.h
    distribution.h
    gauss_distribution.h
    binomial_distribution.h
    roc.h
)

set(SOURCES
    logreg.cpp
    multinomial_logreg.cpp
    knn.cpp
    decision_tree.cpp
    decision_tree_node.cpp
    random_forest.cpp
    naive_bayes.cpp
    gauss_distribution.cpp
    binomial_distribution.cpp
    roc.cpp
)

add_library(mlpp_classification ${HEADER} ${SOURCES})

target_link_libraries(mlpp_classification mlpp_core xtensor-blas)
target_link_libraries(mlpp_classification lapack openblas)
target_include_directories(mlpp_classification PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")