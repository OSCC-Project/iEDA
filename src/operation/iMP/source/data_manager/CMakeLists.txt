set(IMP_PARSER ${IMP_DATAMANAGER}/parser)
set(IMP_BASIC ${IMP_DATAMANAGER}/basic)
add_subdirectory(${IMP_BASIC})
add_subdirectory(${IMP_PARSER})

add_library(imp-datamanager INTERFACE
)

target_link_libraries(imp-datamanager
    INTERFACE
    imp-parser
    imp-basic
)
target_include_directories(imp-datamanager
    INTERFACE
        ${IMP_BASIC}
        ${IMP_PARSER}
)