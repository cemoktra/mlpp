cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(TrainTestSplit traintestsplit.cpp)
target_link_libraries(TrainTestSplit MachineLearning)

add_executable(KFold kfold.cpp)
target_link_libraries(KFold MachineLearning)

add_executable(MultiVar multivar.cpp)
target_link_libraries(MultiVar MachineLearning)

add_executable(Polynomial polynomial.cpp)
target_link_libraries(Polynomial MachineLearning)

add_executable(LogReg logistic_regression.cpp)
target_link_libraries(LogReg MachineLearning)

add_executable(MultiClass multi_class.cpp)
target_link_libraries(MultiClass MachineLearning)