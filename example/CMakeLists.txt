cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(TrainTestSplit traintestsplit.cpp)
target_link_libraries(TrainTestSplit LinearRegression)

add_executable(KFold kfold.cpp)
target_link_libraries(KFold LinearRegression)