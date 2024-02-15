set(MultilevelCoarseningSources
        deterministic_multilevel_coarsener.cpp
        multilevel_uncoarsener.cpp)

set(NLevelCoarseningSources
        nlevel_uncoarsener.cpp)

foreach(modtarget IN LISTS PARTITIONING_SUITE_TARGETS)
    target_sources(${modtarget} PRIVATE ${MultilevelCoarseningSources})
    target_sources(${modtarget} PRIVATE ${NLevelCoarseningSources})
endforeach()