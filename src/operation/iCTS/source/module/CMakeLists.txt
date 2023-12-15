add_subdirectory(${ICTS_MODULE}/evaluator)
add_subdirectory(${ICTS_MODULE}/router)

add_library(icts_module INTERFACE)

target_link_libraries(icts_module INTERFACE icts_evaluator icts_router icts_api
                                            icts_source_external_libs)
