# Define GLOG_USE_GLOG_EXPORT for glog 0.7.1+ compatibility
add_definitions(-DGLOG_USE_GLOG_EXPORT)

add_library(ipl-module-logger INTERFACE)

target_link_directories(ipl-module-logger
    INTERFACE
        ipl-bridge
        log
) 

target_include_directories(ipl-module-logger INTERFACE ${iPL_LOGGER})
target_include_directories(ipl-module-logger INTERFACE ${HOME_UTILITY})