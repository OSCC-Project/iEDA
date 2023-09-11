set(ICTS_DATA_MANAGER ${ICTS_SOURCE}/data_manager)
set(ICTS_MODULE ${ICTS_SOURCE}/module)
set(ICTS_SOLVER ${ICTS_SOURCE}/solver)
set(ICTS_MODEL ${ICTS_SOURCE}/model)

# build
add_subdirectory(${ICTS_DATA_MANAGER})
add_subdirectory(${ICTS_MODULE})
add_subdirectory(${ICTS_SOLVER})
add_subdirectory(${ICTS_MODEL})

add_library(icts_source INTERFACE)

target_link_libraries(icts_source INTERFACE icts_data_manager icts_module
                                            icts_solver icts_model)

target_include_directories(icts_source INTERFACE ${ICTS_SOURCE} ${ICTS_MODEL})
