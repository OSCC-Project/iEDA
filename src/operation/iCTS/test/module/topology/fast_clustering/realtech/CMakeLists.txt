if(ICTS_BUILD_REALTECH_TESTS AND ICTS_BUILD_SLOW_REALTECH_TESTS)
  icts_add_test_executable(
    icts_test_module_topology_fast_clustering_realtech_benchmark
    REALTECH
    SOURCES
      ${ICTS_TEST}/module/topology/fast_clustering/realtech/FastClusteringRealTechBenchmarkArtifacts.cc
      ${ICTS_TEST}/module/topology/fast_clustering/realtech/FastClusteringRealTechBenchmarkDiscovery.cc
      ${ICTS_TEST}/module/topology/fast_clustering/realtech/FastClusteringRealTechBenchmarkMetrics.cc
      ${ICTS_TEST}/module/topology/fast_clustering/realtech/FastClusteringRealTechBenchmarkSetup.cc
      ${ICTS_TEST}/module/topology/fast_clustering/realtech/FastClusteringRealTechBenchmarkTest.cc
    LIBS
      icts_source_module_topology
      icts_test_common_clustering_artifact
      icts_test_common_scoped_log_file
      icts_test_common_realtech_setup
      icts_test_common_visualization
  )
endif()
