add_library(ipl-bridge INTERFACE)

target_link_libraries(ipl-bridge 
    INTERFACE 
        ipl-external_api
        ipl-source_external_libs
)

add_library(ipl-utility 
            Geometry.cc)

target_link_libraries(ipl-utility 
    PUBLIC
        ipl-source_external_libs
)

if(BUILD_QT)
    find_package(Qt5 COMPONENTS Core Widgets REQUIRED)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)
    add_library(ipl-qt Image.cc)
    target_link_libraries(ipl-qt 
        PUBLIC
        Qt5::Core
        Qt5::Widgets
    )
    target_link_libraries(ipl-utility
        PUBLIC
        ipl-qt
    )
endif()   