# Define GLOG_USE_GLOG_EXPORT for glog 0.7.1+ compatibility
add_definitions(-DGLOG_USE_GLOG_EXPORT)

add_library(ipl-module-logger Log.cc)

target_link_directories(ipl-module-logger
    PRIVATE
        ipl-bridge
) 

target_include_directories(ipl-module-logger PUBLIC ${iPL_LOGGER})