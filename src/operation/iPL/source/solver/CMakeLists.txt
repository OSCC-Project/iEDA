set(CONJUGATE_SOLVER ${iPL_SOLVER}/conjugate)
set(NESTEROV_SOLVER ${iPL_SOLVER}/nesterov)
set(SUBGRADIENT_SOLVER ${iPL_SOLVER}/subgradient)
set(SIMULATE_ANNEAL_SOLVER ${iPL_SOLVER}/simulate_anneal)
set(PARTITION_SOLVER ${iPL_SOLVER}/partition)

add_subdirectory(${CONJUGATE_SOLVER})
add_subdirectory(${NESTEROV_SOLVER})
add_subdirectory(${SUBGRADIENT_SOLVER})
add_subdirectory(${SIMULATE_ANNEAL_SOLVER})
add_subdirectory(${PARTITION_SOLVER})

add_library(ipl-solver INTERFACE)

target_link_libraries(ipl-solver
    INTERFACE
        # ipl-solver-conjugate
        ipl-solver-nesterov
        ipl-solver-simulate_anneal
        ipl-solver-partition
        # ipl-solver-subgradient
)

target_include_directories(ipl-solver INTERFACE ${iPL_SOLVER})
