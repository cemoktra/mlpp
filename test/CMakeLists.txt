cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    linreg.cpp
    main.cpp
)

add_executable(UnitTests ${SOURCES})
target_link_libraries(UnitTests gtest_main mlpp_regression)
add_test(NAME unit_tests COMMAND UnitTests)