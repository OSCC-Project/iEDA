add_subdirectory(${ICTS_SOLVER}/BEAT)
add_subdirectory(${ICTS_SOLVER}/GOCA)
add_subdirectory(${ICTS_SOLVER}/polygon)

add_library(icts_solver INTERFACE)

target_link_libraries(icts_solver INTERFACE icts_data_manager icts_polygon
                                            icts_beat icts_goca icts_api)

target_include_directories(
  icts_solver INTERFACE ${ICTS_SOLVER}/BEAT ${ICTS_SOLVER}/GOCA
                        ${ICTS_SOLVER}/polygon)
