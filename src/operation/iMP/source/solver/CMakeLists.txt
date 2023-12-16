add_library(imp-solver 
           ${IMP_SOLVER}/Annealer.cc
           ${IMP_SOLVER}/Partitionner.cc)


target_link_libraries(imp-solver 
                      PUBLIC
                      imp-logger)

target_include_directories(imp-solver 
                           PUBLIC 
                           ${IMP_SOLVER})

set(HMETIS_BINARY ${PROJECT_SOURCE_DIR}/src/third_party/hmetis/hmetis2.0pre1)
message(STATUS "MP: Hmetis binary path - " ${HMETIS_BINARY})
add_definitions("-DHMETIS_BINARY=\"${HMETIS_BINARY}\"")