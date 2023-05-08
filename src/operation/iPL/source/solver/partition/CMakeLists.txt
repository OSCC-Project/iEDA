add_library(ipl-solver-partition Partition.cc)
target_link_libraries(ipl-solver-partition ${iPL_SOURCE}/module/macro_placer/libs/libmetis.a)

target_include_directories(ipl-solver-partition PUBLIC ${PARTITION_SOLVER})