add_subdirectory(fixture)

icts_add_test_executable(
  icts_test_module_topology_gen
  PUBLIC_RUNTIME
  SOURCES
    ${ICTS_TEST}/module/topology/topology_gen/TopologyGenTest.cc
    ${ICTS_TEST}/module/topology/topology_gen/TopologyGenDepthTest.cc
  LIBS
    icts_test_module_topology_gen_fixture
)
