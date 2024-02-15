if ( KAHYPAR_ENABLE_STEINER_TREE_METRIC )
  target_sources(mt_kahypar_tests PRIVATE
          target_graph_test.cc
          set_enumerator_test.cc
          )
endif()
