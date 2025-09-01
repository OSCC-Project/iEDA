set(WIRELENGTH_SOURCES
    HPWirelength.cc
    WAWirelengthGradient.cc
    SteinerWirelength.cc
)

if(BUILD_AI_PREDICTOR)
    list(APPEND WIRELENGTH_SOURCES AIWirelength.cc)
endif()

add_library(ipl_module_evaluator_wirelength ${WIRELENGTH_SOURCES})


target_link_libraries(ipl_module_evaluator_wirelength
    PUBLIC
        ipl-module-topology_manager
        ipl-module-grid_manager
    PRIVATE
        ipl-module-logger
        ipl-configurator
        ipl-bridge
)

if(BUILD_AI_PREDICTOR)
    target_link_libraries(ipl_module_evaluator_wirelength
        PUBLIC
            ipl-module-ai_predictor
    )
endif()

target_include_directories(ipl_module_evaluator_wirelength 
    PUBLIC 
        ${iPL_WIRELENGTH_EVALUATOR}
)

if(BUILD_AI_PREDICTOR)
    target_include_directories(ipl_module_evaluator_wirelength 
        PUBLIC 
            ${iPL_AI_PREDICTOR}/wirelength
    )
endif()