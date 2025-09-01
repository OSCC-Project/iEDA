add_library(ipl_module_ai_wirelength
    wirelength_predictor.cc
    onnx_model_handler.cc
    normalization_handler.cc
)

target_link_libraries(ipl_module_ai_wirelength
    PUBLIC
    ${HOME_THIRDPARTY}/onnxruntime/libonnxruntime.so
)

target_include_directories(ipl_module_ai_wirelength
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HOME_THIRDPARTY}/onnxruntime/include
    ${HOME_THIRDPARTY}/json
)