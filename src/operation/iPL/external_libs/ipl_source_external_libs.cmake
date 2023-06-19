add_library(ipl-source_external_libs INTERFACE)
find_package(Qt5 COMPONENTS Core Widgets REQUIRED)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
target_link_libraries(ipl-source_external_libs
    INTERFACE
        fftsg_library
        flute
        idb
        IdbBuilder
        def_builder
        lef_builder
        def_service
        lef_service
        usage
        Qt5::Core
        Qt5::Widgets
)

target_include_directories(ipl-source_external_libs
    INTERFACE
        ${HOME_DATABASE}/db
        ${HOME_DATABASE}/builder/builder
        ${HOME_DATABASE}/builder/def_builder/def_service
        ${HOME_DATABASE}/builder/lef_builder/lef_service
        ${HOME_UTILITY}
)

target_include_directories(ipl-source_external_libs
    SYSTEM INTERFACE
        ${HOME_THIRDPARTY}/flute3
        ${HOME_THIRDPARTY}/json
        ${HOME_THIRDPARTY}/fft
)