add_library(ipl-random_placer RandomPlace.cc)

target_link_libraries(ipl-random_placer
    PUBLIC
        ipl-module-wrapper
        ipl-module-logger
        ipl-configurator
)

target_include_directories(ipl-random_placer PUBLIC ${iPL_RANDOM_PLACER})