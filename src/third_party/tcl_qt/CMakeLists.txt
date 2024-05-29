set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_PREFIX_PATH "/opt/Qt/5.15.2/gcc_64")
find_package(Qt5 COMPONENTS Core REQUIRED)
include_directories(${QT_INCLUDE})
add_library(tcl_qt tcl_qt.cpp )
target_link_libraries( tcl_qt Qt5${QT_VERSION_MAJOR}::Core )

find_package(TCL 8.6 QUIET)
if(TCL_FOUND)
  target_include_directories(tcl_qt PUBLIC ${TCL_INCLUDE_PATH})
else(TCL_FOUND)
  message(FATAL_ERROR "Could not find TCL!")
endif(TCL_FOUND)

target_include_directories(tcl_qt
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}       
)