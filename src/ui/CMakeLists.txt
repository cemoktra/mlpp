cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    main.cpp
)

set(HEADERS
)

set(RESOURCES
  qml.qrc
)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

if (WIN32)
  set(Qt5_DIR "C:/Qt/5.13.1/mingw73_64/lib/cmake/Qt5")
endif()
find_package(Qt5 COMPONENTS Core Quick Widgets QuickControls2 QuickTemplates2 REQUIRED)

add_executable(mlpp_ui ${SOURCES} ${HEADERS} ${RESOURCES})
target_compile_definitions(mlpp_ui
  PRIVATE $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:QT_QML_DEBUG>)
target_link_libraries(mlpp_ui
  PRIVATE Qt5::Core Qt5::Quick Qt5::Gui Qt5::Qml Qt5::Network Qt5::Widgets Qt5::QuickControls2 Qt5::QuickTemplates2 tbb)


if (WIN32)
  add_custom_command(TARGET mlpp_ui POST_BUILD 
    COMMAND "${CMAKE_COMMAND}" -E copy $<TARGET_FILE:Qt5::Core> $<TARGET_FILE_DIR:mlpp_ui>
    COMMAND "${CMAKE_COMMAND}" -E copy $<TARGET_FILE:Qt5::Gui> $<TARGET_FILE_DIR:mlpp_ui>
    COMMAND "${CMAKE_COMMAND}" -E copy $<TARGET_FILE:Qt5::Quick> $<TARGET_FILE_DIR:mlpp_ui>
    COMMAND "${CMAKE_COMMAND}" -E copy $<TARGET_FILE:Qt5::Qml> $<TARGET_FILE_DIR:mlpp_ui>
    COMMAND "${CMAKE_COMMAND}" -E copy $<TARGET_FILE:Qt5::Network> $<TARGET_FILE_DIR:mlpp_ui>
    COMMAND "${CMAKE_COMMAND}" -E copy $<TARGET_FILE:Qt5::Widgets> $<TARGET_FILE_DIR:mlpp_ui>
    COMMAND "${CMAKE_COMMAND}" -E copy $<TARGET_FILE:Qt5::QuickControls2> $<TARGET_FILE_DIR:mlpp_ui>
    COMMAND "${CMAKE_COMMAND}" -E copy $<TARGET_FILE:Qt5::QuickTemplates2> $<TARGET_FILE_DIR:mlpp_ui>
  )  
endif()
