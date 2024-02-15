target_sources(mt_kahypar_tests PRIVATE
        static_hypergraph_test.cc
        partitioned_hypergraph_test.cc
        delta_partitioned_hypergraph_test.cc
        partitioned_hypergraph_smoke_test.cc
        gain_updates_test.cc
        graph_test.cc
        connectivity_set_test.cc
        priority_queue_test.cc
        array_test.cc
        sparse_map_test.cc
        pin_count_in_part_test.cc
        static_bitset_test.cc
        fixed_vertex_support_test.cc)

if ( KAHYPAR_ENABLE_GRAPH_PARTITIONING_FEATURES )
  target_sources(mt_kahypar_tests PRIVATE
          static_graph_test.cc
          partitioned_graph_test.cc
          delta_partitioned_graph_test.cc)
endif()

if ( KAHYPAR_ENABLE_HIGHEST_QUALITY_FEATURES )
  target_sources(mt_kahypar_tests PRIVATE
          dynamic_hypergraph_test.cc
          dynamic_partitioned_hypergraph_test.cc
          contraction_tree_test.cc
          incident_net_array_test.cc
          nlevel_smoke_test.cc)

  if ( KAHYPAR_ENABLE_GRAPH_PARTITIONING_FEATURES )
    target_sources(mt_kahypar_tests PRIVATE
      dynamic_adjacency_array_test.cc
      dynamic_graph_test.cc
      nlevel_smoke_test.cc)
  endif()
endif()
