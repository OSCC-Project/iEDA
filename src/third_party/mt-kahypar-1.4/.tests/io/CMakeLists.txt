configure_file(context_test.cc.in ${PROJECT_BINARY_DIR}/tests/io/context_test.cc)
target_sources(mt_kahypar_tests PRIVATE
        hypergraph_io_test.cc
        ${PROJECT_BINARY_DIR}/tests/io/context_test.cc
        sql_plottools_serializer_test.cc
        )