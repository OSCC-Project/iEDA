#set
set(iPL_LAYOUT_CHECKER ${iPL_CHECKER}/layout_checker)

add_subdirectory(${iPL_LAYOUT_CHECKER})

add_library(ipl-module-checker INTERFACE)

target_link_libraries(ipl-module-checker
    INTERFACE
        ipl-layout_checker
)

target_include_directories(ipl-module-checker INTERFACE ${iPL_CHECKER})