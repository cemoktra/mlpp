cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ex_regression regression.cpp)
target_link_libraries(ex_regression mlpp_regression)

add_executable(ex_classification classification.cpp)
target_link_libraries(ex_classification mlpp_classification pthread)

# add_executable(ex_curves curves.cpp)
# target_link_libraries(ex_curves mlpp_classification)

add_executable(ex_spam spamfilter.cpp)
target_link_libraries(ex_spam mlpp_classification)

add_executable(ex_mnist mnist.cpp)
target_link_libraries(ex_mnist mlpp_classification)

if (WIN32)
  add_custom_command(TARGET ex_mnist POST_BUILD 
    COMMAND "${CMAKE_COMMAND}" -E copy "${CONDA_LIB_BIN}/zlib.dll" $<TARGET_FILE_DIR:ex_mnist>
  )  
endif()