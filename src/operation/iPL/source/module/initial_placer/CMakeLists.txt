set(iPL_CENTER_PLACER ${iPL_IP}/center_placer)
set(iPL_RANDOM_PLACER ${iPL_IP}/random_placer)

add_subdirectory(${iPL_CENTER_PLACER})
add_subdirectory(${iPL_RANDOM_PLACER})

add_library(ipl-module-initial_placer INTERFACE)

target_link_libraries(ipl-module-initial_placer 
    INTERFACE
        ipl-center_placer
        ipl-random_placer
)

target_include_directories(ipl-module-initial_placer INTERFACE ${iPL_IP})