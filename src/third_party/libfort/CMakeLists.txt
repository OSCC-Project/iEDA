option(FORT_HAVE_WCHAR "Enable wchar support" ON)
option(FORT_HAVE_UTF8 "Enable UTF8 support" ON)

add_library(fort
    fort.c)
add_library(${PROJECT_NAME}::fort ALIAS fort)

target_include_directories(fort
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set_target_properties(fort
    PROPERTIES
        VERSION ${libfort_VERSION}
        SOVERSION ${libfort_SOVERSION}
        POSITION_INDEPENDENT_CODE ON
)

if(NOT FORT_HAVE_WCHAR)
    target_compile_definitions(fort
        PRIVATE
            -DFT_CONGIG_DISABLE_WCHAR
    )
endif()

if(NOT FORT_HAVE_UTF8)
    target_compile_definitions(fort
        PRIVATE
            -DFT_CONGIG_DISABLE_UTF8
    )
endif()

