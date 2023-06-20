add_library(ipl-bridge INTERFACE)

target_link_libraries(ipl-bridge 
    INTERFACE 
        ipl-api
        ipl-source_external_libs
)

add_library(ipl-utility 
            Image.cc
            Geometry.cc)

target_link_libraries(ipl-utility 
    PUBLIC
        ipl-source_external_libs
)