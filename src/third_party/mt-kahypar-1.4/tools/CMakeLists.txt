add_executable(SnapGraphToHgr snap_graph_to_hgr.cc)
target_link_libraries(SnapGraphToHgr ${Boost_LIBRARIES})
target_link_libraries(SnapGraphToHgr TBB::tbb TBB::tbbmalloc_proxy)
set_property(TARGET SnapGraphToHgr PROPERTY CXX_STANDARD 17)
set_property(TARGET SnapGraphToHgr PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(GraphToHgr graph_to_hgr.cc)
target_link_libraries(GraphToHgr ${Boost_LIBRARIES})
target_link_libraries(GraphToHgr TBB::tbb TBB::tbbmalloc_proxy)
set_property(TARGET GraphToHgr PROPERTY CXX_STANDARD 17)
set_property(TARGET GraphToHgr PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(HgrToGraph hgr_to_graph.cc)
target_link_libraries(HgrToGraph ${Boost_LIBRARIES})
target_link_libraries(HgrToGraph TBB::tbb TBB::tbbmalloc_proxy)
set_property(TARGET HgrToGraph PROPERTY CXX_STANDARD 17)
set_property(TARGET HgrToGraph PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(HgrToParkway hgr_to_parkway_converter.cc)
target_link_libraries(HgrToParkway ${Boost_LIBRARIES})
target_link_libraries(HgrToParkway TBB::tbb TBB::tbbmalloc_proxy)
set_property(TARGET HgrToParkway PROPERTY CXX_STANDARD 17)
set_property(TARGET HgrToParkway PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(HgrToZoltan hgr_to_zoltan_converter.cc)
target_link_libraries(HgrToZoltan ${Boost_LIBRARIES})
target_link_libraries(HgrToZoltan TBB::tbb TBB::tbbmalloc_proxy)
set_property(TARGET HgrToZoltan PROPERTY CXX_STANDARD 17)
set_property(TARGET HgrToZoltan PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(HypergraphStats hypergraph_stats.cc)
target_link_libraries(HypergraphStats ${Boost_LIBRARIES})
target_link_libraries(HypergraphStats TBB::tbb TBB::tbbmalloc_proxy)
set_property(TARGET HypergraphStats PROPERTY CXX_STANDARD 17)
set_property(TARGET HypergraphStats PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(MetisToScotch metis_to_scotch_converter.cc)
target_link_libraries(MetisToScotch ${Boost_LIBRARIES})
target_link_libraries(MetisToScotch TBB::tbb TBB::tbbmalloc_proxy)
set_property(TARGET MetisToScotch PROPERTY CXX_STANDARD 17)
set_property(TARGET MetisToScotch PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(OneToOneMapping one_to_one_mapping.cc)
target_link_libraries(OneToOneMapping ${Boost_LIBRARIES})
target_link_libraries(OneToOneMapping TBB::tbb TBB::tbbmalloc_proxy)
set_property(TARGET OneToOneMapping PROPERTY CXX_STANDARD 17)
set_property(TARGET OneToOneMapping PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(VerifyTargetGraphPartition verify_process_mapping_partition.cc)
target_link_libraries(VerifyTargetGraphPartition ${Boost_LIBRARIES})
target_link_libraries(VerifyTargetGraphPartition TBB::tbb TBB::tbbmalloc_proxy)
set_property(TARGET VerifyTargetGraphPartition PROPERTY CXX_STANDARD 17)
set_property(TARGET VerifyTargetGraphPartition PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(GridGraphGenerator grid_graph_generator.cc)
target_link_libraries(GridGraphGenerator ${Boost_LIBRARIES})
target_link_libraries(GridGraphGenerator TBB::tbb TBB::tbbmalloc_proxy)
set_property(TARGET GridGraphGenerator PROPERTY CXX_STANDARD 17)
set_property(TARGET GridGraphGenerator PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(HierarchicalTargetGraphGenerator hierarchical_process_graph_generator.cc)
target_link_libraries(HierarchicalTargetGraphGenerator ${Boost_LIBRARIES})
target_link_libraries(HierarchicalTargetGraphGenerator TBB::tbb TBB::tbbmalloc_proxy)
set_property(TARGET HierarchicalTargetGraphGenerator PROPERTY CXX_STANDARD 17)
set_property(TARGET HierarchicalTargetGraphGenerator PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(SnapToMetis snap_to_metis_converter.cc)
target_link_libraries(SnapToMetis ${Boost_LIBRARIES})
target_link_libraries(SnapToMetis TBB::tbb TBB::tbbmalloc_proxy)
set_property(TARGET SnapToMetis PROPERTY CXX_STANDARD 17)
set_property(TARGET SnapToMetis PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(EvaluateBipart evaluate_bipart_partition.cc)
target_link_libraries(EvaluateBipart ${Boost_LIBRARIES})
target_link_libraries(EvaluateBipart TBB::tbb TBB::tbbmalloc_proxy)
set_property(TARGET EvaluateBipart PROPERTY CXX_STANDARD 17)
set_property(TARGET EvaluateBipart PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(EvaluatePartition evaluate_hmetis_partition.cc)
target_link_libraries(EvaluatePartition ${Boost_LIBRARIES})
set_property(TARGET EvaluatePartition PROPERTY CXX_STANDARD 17)
set_property(TARGET EvaluatePartition PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(VerifyPartition verify_partition.cc)
target_link_libraries(VerifyPartition ${Boost_LIBRARIES})
target_link_libraries(VerifyPartition TBB::tbb TBB::tbbmalloc_proxy)
set_property(TARGET VerifyPartition PROPERTY CXX_STANDARD 17)
set_property(TARGET VerifyPartition PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(FixedVertexFileGenerator fixed_vertex_file_generator.cc)
target_link_libraries(FixedVertexFileGenerator ${Boost_LIBRARIES})
target_link_libraries(FixedVertexFileGenerator TBB::tbb TBB::tbbmalloc_proxy)
set_property(TARGET FixedVertexFileGenerator PROPERTY CXX_STANDARD 17)
set_property(TARGET FixedVertexFileGenerator PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(BenchShuffle bench_deterministic_shuffling.cpp bench_deterministic_shuffling.cpp)
target_link_libraries(BenchShuffle TBB::tbb TBB::tbbmalloc_proxy)
set_property(TARGET BenchShuffle PROPERTY CXX_STANDARD 17)
set_property(TARGET BenchShuffle PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(MtxToGraph mtx_to_graph.cc)
set_property(TARGET MtxToGraph PROPERTY CXX_STANDARD 17)
set_property(TARGET MtxToGraph PROPERTY CXX_STANDARD_REQUIRED ON)

set(TOOLS_TARGETS ${TOOLS_TARGETS} GraphToHgr
                                   HgrToGraph
                                   EvaluateBipart
                                   VerifyPartition
                                   EvaluatePartition
                                   HgrToParkway
                                   HgrToZoltan
                                   HypergraphStats
                                   MetisToScotch
                                   OneToOneMapping
                                   VerifyTargetGraphPartition
                                   GridGraphGenerator
                                   HierarchicalTargetGraphGenerator
                                   FixedVertexFileGenerator
                                   PARENT_SCOPE)
