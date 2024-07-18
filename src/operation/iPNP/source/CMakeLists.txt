add_subdirectory(config)
add_subdirectory(data_manager)
add_subdirectory(module)
add_subdirectory(utility)

add_library(pnp
iPNP.cpp
)

target_link_libraries(pnp optimizer synthesis data_manager idb tcl usage pthread stdc++fs IdbBuilder def_service lef_service)