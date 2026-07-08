add_subdirectory(case)
add_subdirectory(analysis)
add_subdirectory(artifact)
add_subdirectory(core)

add_library(icts_test_module_topology_gen_fixture INTERFACE)

target_link_libraries(
  icts_test_module_topology_gen_fixture
  INTERFACE
  icts_test_module_topology_gen_scenario_runner
)
