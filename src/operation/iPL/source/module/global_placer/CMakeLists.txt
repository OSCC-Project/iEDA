# set
set(iPL_ELECTROSTATIC_PLACER 
    ${iPL_GP}/electrostatic_placer
)

add_subdirectory(${iPL_ELECTROSTATIC_PLACER})

add_library(ipl-module-global_placer INTERFACE)

target_link_libraries(ipl-module-global_placer
    INTERFACE
        ipl-electrostatic_placer
)

target_include_directories(ipl-module-global_placer INTERFACE ${iPL_GP})