#include <memory>
#include <vector>
#include <iostream>
#include <thread>

#include <libmtkahypar.h>

// Install library interface via 'sudo make install.mtkahypar' in build folder
// Compile with: g++ -std=c++14 -DNDEBUG -O3 map_hypergraph_onto_target_graph.cc -o example -lmtkahypar
int main(int argc, char* argv[]) {

  // Initialize thread pool
  mt_kahypar_initialize_thread_pool(
    std::thread::hardware_concurrency() /* use all available cores */,
    true /* activate interleaved NUMA allocation policy */ );

  // Setup partitioning context
  mt_kahypar_context_t* context = mt_kahypar_context_new();
  mt_kahypar_load_preset(context, DEFAULT /* corresponds to MT-KaHyPar-D */);
  // In the following, we map a hypergraph into target graph with 8 nodes
  // with an allowed imbalance of 3%
  mt_kahypar_set_partitioning_parameters(context,
    8 /* number of blocks */, 0.03 /* imbalance parameter */,
    KM1 /* objective function - not relevant for mapping */);
  mt_kahypar_set_seed(42 /* seed */);
  // Enable logging
  mt_kahypar_set_context_parameter(context, VERBOSE, "1");

  // Load Hypergraph for DEFAULT preset
  mt_kahypar_hypergraph_t hypergraph =
    mt_kahypar_read_hypergraph_from_file("ibm01.hgr",
      DEFAULT, HMETIS /* file format */);

  // Read target graph file in Metis file format
  mt_kahypar_target_graph_t* target_graph =
    mt_kahypar_read_target_graph_from_file("target.graph");

  // Map hypergraph onto target graph
  mt_kahypar_partitioned_hypergraph_t partitioned_hg =
    mt_kahypar_map(hypergraph, target_graph, context);

  // Extract Mapping
  std::unique_ptr<mt_kahypar_partition_id_t[]> mapping =
    std::make_unique<mt_kahypar_partition_id_t[]>(mt_kahypar_num_hypernodes(hypergraph));
  mt_kahypar_get_partition(partitioned_hg, mapping.get());

  // Extract Block Weights
  std::unique_ptr<mt_kahypar_hypernode_weight_t[]> block_weights =
    std::make_unique<mt_kahypar_hypernode_weight_t[]>(8);
  mt_kahypar_get_block_weights(partitioned_hg, block_weights.get());

  // Compute Metrics
  const double imbalance = mt_kahypar_imbalance(partitioned_hg, context);
  const double steiner_tree_metric = mt_kahypar_steiner_tree(partitioned_hg, target_graph);

  // Output Results
  std::cout << "Partitioning Results:" << std::endl;
  std::cout << "Imbalance           = " << imbalance << std::endl;
  std::cout << "Steiner Tree Metric = " << steiner_tree_metric << std::endl;
  for ( size_t i = 0; i < 8; ++i ) {
    std::cout << "Weight of Block " << i << "   = " << block_weights[i] << std::endl;
  }

  mt_kahypar_free_context(context);
  mt_kahypar_free_hypergraph(hypergraph);
  mt_kahypar_free_partitioned_hypergraph(partitioned_hg);
  mt_kahypar_free_target_graph(target_graph);
}