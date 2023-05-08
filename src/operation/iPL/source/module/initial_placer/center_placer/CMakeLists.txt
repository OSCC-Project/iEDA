add_library(ipl-center_placer CenterPlace.cc)

target_link_libraries(ipl-center_placer
    PUBLIC
        ipl-module-wrapper
        ipl-module-logger
        ipl-configurator
)

target_include_directories(ipl-center_placer PUBLIC ${iPL_CENTER_PLACER})