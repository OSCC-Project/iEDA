add_library(ipl_predictor_wirelen
    ai_wirelength.cc
    wirelength_predictor.cc
    onnx_model_handler.cc
    normalization_handler.cc
)

target_link_libraries(ipl_predictor_wirelen 
    PUBLIC
    ${HOME_THIRDPARTY}/onnxruntime/libonnxruntime.so
    log
    ipl_module_evaluator_wirelength
    ipl-module-topology_manager
)

target_include_directories(ipl_predictor_wirelen 
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_THIRDPARTY}/onnxruntime/include
    ${HOME_THIRDPARTY}/json
    ${HOME_OPERATION}/iPL/source/data
)
