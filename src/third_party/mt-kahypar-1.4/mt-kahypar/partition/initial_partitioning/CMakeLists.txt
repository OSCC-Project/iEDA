set(InitialPartitioningSources
        pool_initial_partitioner.cpp
        random_initial_partitioner.cpp
        bfs_initial_partitioner.cpp
        label_propagation_initial_partitioner.cpp
        )

foreach(modtarget IN LISTS PARTITIONING_SUITE_TARGETS)
    target_sources(${modtarget} PRIVATE ${InitialPartitioningSources})
endforeach()