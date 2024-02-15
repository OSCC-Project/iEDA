add_subdirectory(partition)
add_subdirectory(datastructures)
add_subdirectory(io)
add_subdirectory(utils)

foreach(modtarget IN LISTS PARTITIONING_SUITE_TARGETS)
    target_precompile_headers(${modtarget} PRIVATE definitions.h)
    target_precompile_headers(${modtarget} PRIVATE partition/context.h)
endforeach()