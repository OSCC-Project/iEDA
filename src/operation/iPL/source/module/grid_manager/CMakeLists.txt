add_library(ipl-module-grid_manager 
                GridManager.cc
                )

target_link_libraries(ipl-module-grid_manager
    PUBLIC
        ipl-module-logger
        ipl-utility
)

target_include_directories(ipl-module-grid_manager PUBLIC ${iPL_GRID_MANAGER})