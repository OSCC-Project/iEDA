set(UtilSources
      memory_tree.cpp
    )

foreach(modtarget IN LISTS PARTITIONING_SUITE_TARGETS)
    target_sources(${modtarget} PRIVATE ${UtilSources})
endforeach()

foreach(modtarget IN LISTS TOOLS_TARGETS)
    target_sources(${modtarget} PRIVATE ${UtilSources})
endforeach()