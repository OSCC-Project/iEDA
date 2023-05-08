set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt5 COMPONENTS Core REQUIRED)
include_directories(${QT_INCLUDE})
add_library(tcl_qt tcl_qt.cpp )
target_link_libraries( tcl_qt Qt${QT_VERSION_MAJOR}::Core )

target_include_directories(tcl_qt
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}       
)