AUX_SOURCE_DIRECTORY(./ TCL_FEATURE_SRC)

add_library(tcl_feature 
    ${TCL_FEATURE_SRC}
)

target_link_libraries(tcl_feature 
    PUBLIC
        tcl
        shell-cmd
        ieda_feature
)

target_include_directories(tcl_feature
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

