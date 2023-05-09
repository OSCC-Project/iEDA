add_library(ipl-solver-partition Metis.cc Hmetis.cc)
target_link_libraries(ipl-solver-partition ${iPL_SOURCE}/module/macro_placer/libs/libmetis.a)
