# set(iPL_MP ${iPL_MODULE}/macro_placer)
set(iPL_IP ${iPL_MODULE}/initial_placer)
set(iPL_GP ${iPL_MODULE}/global_placer)
set(iPL_POSTGP ${iPL_MODULE}/post_global_placer)
set(iPL_LG ${iPL_MODULE}/legalizer)
set(iPL_DP ${iPL_MODULE}/detail_placer)
set(iPL_EVAL ${iPL_MODULE}/evaluator)
set(iPL_CHECKER ${iPL_MODULE}/checker)
set(iPL_BUFFER ${iPL_MODULE}/buffer)
set(iPL_FILLER ${iPL_MODULE}/filler)
set(iPL_GRID_MANAGER ${iPL_MODULE}/grid_manager)
set(iPL_LOGGER ${iPL_MODULE}/logger)
set(iPL_TOPOLOGY_MANAGER ${iPL_MODULE}/topology_manager)
set(iPL_WRAPPER ${iPL_MODULE}/wrapper)
set(iPL_AI_PREDICTOR ${iPL_MODULE}/ai_predictor)

# add_subdirectory(${iPL_MP})
add_subdirectory(${iPL_IP})
add_subdirectory(${iPL_GP})
add_subdirectory(${iPL_POSTGP})
add_subdirectory(${iPL_LG})
add_subdirectory(${iPL_DP})
add_subdirectory(${iPL_EVAL})
add_subdirectory(${iPL_CHECKER})
add_subdirectory(${iPL_BUFFER})
add_subdirectory(${iPL_FILLER})
add_subdirectory(${iPL_GRID_MANAGER})
add_subdirectory(${iPL_LOGGER})
add_subdirectory(${iPL_TOPOLOGY_MANAGER})
add_subdirectory(${iPL_WRAPPER})
add_subdirectory(${iPL_AI_PREDICTOR})

add_library(ipl-module INTERFACE)
target_link_libraries(ipl-module
    INTERFACE
    ipl-module-checker
    ipl-module-detail_placer
    ipl-module-evaluator

    ipl-module-buffer
    ipl-module-filler
    ipl-module-global_placer
    ipl-module-post_gp
    ipl-module-grid_manager
    ipl-module-legalizer
    ipl-module-logger
    # ipl-module-macro_placer
    ipl-module-initial_placer
    ipl-module-topology_manager
    ipl-module-wrapper
    ipl-module-ai_predictor
)

target_include_directories(ipl-module INTERFACE ${iPL_MODULE})
