set(iPL_CENTER_PLACER ${iPL_IP}/center_placer)

add_subdirectory(${iPL_CENTER_PLACER})

add_library(ipl-module-initial_placer INTERFACE)

target_link_libraries(ipl-module-initial_placer 
    INTERFACE
        ipl-center_placer
)

target_include_directories(ipl-module-initial_placer INTERFACE ${iPL_IP})