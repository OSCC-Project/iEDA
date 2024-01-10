add_library(imp-basic 
            Netlist.cc
            Block.cc
            )

target_link_libraries(imp-basic
    PUBLIC
    imp-structure
)
target_include_directories(imp-basic 
                          PUBLIC 
                          ${IMP_BASIC})