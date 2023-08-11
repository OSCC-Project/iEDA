add_subdirectory(${ICTS_SOLVER}/clustering)
add_subdirectory(${ICTS_SOLVER}/DME)
add_subdirectory(${ICTS_SOLVER}/HCTS)
add_subdirectory(${ICTS_SOLVER}/GOCA)
add_subdirectory(${ICTS_SOLVER}/polygon)
add_subdirectory(${ICTS_SOLVER}/slewAware)
add_subdirectory(${ICTS_SOLVER}/costCalculator)
add_subdirectory(${ICTS_SOLVER}/timingCalculator)

add_library(icts_solver INTERFACE)

target_link_libraries(
  icts_solver
  INTERFACE icts_data_manager
            icts_clustering
            icts_dme
            icts_polygon
            icts_hcts
            icts_goca
            icts_slew_aware
            icts_cost_calculator
            icts_timing_calculator
            icts_api)

target_include_directories(
  icts_solver
  INTERFACE ${ICTS_SOLVER}/clustering
            ${ICTS_SOLVER}/DME
            ${ICTS_SOLVER}/polygon
            ${ICTS_SOLVER}/HCTS
            ${ICTS_SOLVER}/GOCA
            ${ICTS_SOLVER}/slewAware
            ${ICTS_SOLVER}/costCalculator
            ${ICTS_SOLVER}/timingCalculator)
