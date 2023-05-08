add_library(ipl-utility INTERFACE)

target_link_libraries(ipl-utility 
    INTERFACE 
        ipl-api
        ipl-source_external_libs
)