# set
set(iPL_ANALYTICAL_PLACER 
    ${iPL_GP}/analytical_placer
)

add_subdirectory(${iPL_ANALYTICAL_PLACER})

add_library(ipl-module-global_placer INTERFACE)

target_link_libraries(ipl-module-global_placer
    INTERFACE
        ipl-analytical_placer
)

target_include_directories(ipl-module-global_placer INTERFACE ${iPL_GP})