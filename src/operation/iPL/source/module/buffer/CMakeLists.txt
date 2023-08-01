add_library(ipl-module-buffer BufferInserter.cc)

target_link_libraries(ipl-module-buffer
    PRIVATE
        ipl-configurator
        ipl-bridge
        ipl_module_evaluator_wirelength
        ipl-module-logger
)

target_include_directories(ipl-module-buffer PUBLIC ${iPL_BUFFER})