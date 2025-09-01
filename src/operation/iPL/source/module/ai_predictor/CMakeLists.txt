#set
set(iPL_WIRELENGTH_PREDICT ${iPL_AI_PREDICTOR}/wirelength)

add_subdirectory(${iPL_WIRELENGTH_PREDICT})

add_library(ipl-module-ai_predictor INTERFACE)

target_link_libraries(ipl-module-ai_predictor
    INTERFACE
    ipl_module_ai_wirelength
)

target_include_directories(ipl-module-ai_predictor INTERFACE ${iPL_AI_PREDICTOR})