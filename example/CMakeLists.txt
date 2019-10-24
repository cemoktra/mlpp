cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(LinearRegressionExample main.cpp)
target_link_libraries(LinearRegressionExample LinearRegression)