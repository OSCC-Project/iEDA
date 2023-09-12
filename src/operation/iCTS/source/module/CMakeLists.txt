# add_subdirectory(${ICTS_MODULE}/balancer)
add_subdirectory(${ICTS_MODULE}/evaluator)
# add_subdirectory(${ICTS_MODULE}/optimizer)
add_subdirectory(${ICTS_MODULE}/router)
# add_subdirectory(${ICTS_MODULE}/synthesis)

add_library(icts_module INTERFACE)

target_link_libraries(
  icts_module
  INTERFACE # icts_balancer
            icts_evaluator
            # icts_optimizer
            icts_router
            # icts_synthesis
            icts_api
            icts_source_external_libs)
