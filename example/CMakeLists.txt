cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ex_regression regression.cpp)
target_link_libraries(ex_regression mlpp_regression)

add_executable(ex_classification classification.cpp)
target_link_libraries(ex_classification mlpp_classification)

add_executable(ex_curves curves.cpp)
target_link_libraries(ex_curves mlpp_classification)

add_executable(ex_spam spamfilter.cpp)
target_link_libraries(ex_spam mlpp_classification)