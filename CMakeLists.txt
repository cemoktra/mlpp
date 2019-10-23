project(MachineLearning)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(LinearRegression main.cpp linreg.cpp linreg.h)
target_link_libraries(LinearRegression tbb)