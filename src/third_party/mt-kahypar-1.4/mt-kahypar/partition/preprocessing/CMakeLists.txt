set(PreprocessingSources
        community_detection/parallel_louvain.cpp
        community_detection/local_moving_modularity.cpp)

foreach(modtarget IN LISTS PARTITIONING_SUITE_TARGETS)
    target_sources(${modtarget} PRIVATE ${PreprocessingSources})
endforeach()