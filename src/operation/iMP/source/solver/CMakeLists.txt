add_library(imp-solver 
           ${IMP_SOLVER}/Annealer.cc
           ${IMP_SOLVER}/Partitionner.cc)

target_link_libraries(imp-solver 
                      PRIVATE
                      imp-logger)

target_include_directories(imp-solver 
                           PUBLIC 
                           ${IMP_SOLVER})
