add_library(ipl-layout_checker LayoutChecker.cc)

target_link_libraries(ipl-layout_checker 
    PUBLIC
        ipl-configurator
        ipl-module-logger
        ipl-module-grid_manager
)

target_include_directories(ipl-layout_checker PUBLIC ${iPL_LAYOUT_CHECKER})