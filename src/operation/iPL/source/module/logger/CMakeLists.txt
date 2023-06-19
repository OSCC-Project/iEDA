add_library(ipl-module-logger Log.cc)

target_link_directories(ipl-module-logger
    PRIVATE
        ipl-bridge
) 

target_include_directories(ipl-module-logger PUBLIC ${iPL_LOGGER})