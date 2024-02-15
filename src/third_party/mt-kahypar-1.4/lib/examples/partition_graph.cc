#include <memory>
#include <vector>
#include <iostream>
#include <thread>

#include <libmtkahypar.h>

// Install library interface via 'sudo make install.mtkahypar' in build folder
// Compile with: g++ -std=c++14 -DNDEBUG -O3 partition_graph.cc -o example -lmtkahypar
int main(int argc, char* argv[]) {

  // Initialize thread pool
  mt_kahypar_initialize_thread_pool(
    std::thread::hardware_concurrency() /* use all available cores */,
    true /* activate interleaved NUMA allocation policy */ );

  // Setup partitioning context
  mt_kahypar_context_t* context = mt_kahypar_context_new();
  mt_kahypar_load_preset(context, DEFAULT /* corresponds to MT-KaHyPar-D */);
  // In the following, we partition a graph into two blocks
  // with an allowed imbalance of 3% and optimize the edge cut (CUT)
  mt_kahypar_set_partitioning_parameters(context,
    2 /* number of blocks */, 0.03 /* imbalance parameter */,
    CUT /* objective function */);
  mt_kahypar_set_seed(42 /* seed */);
  // Enable logging
  mt_kahypar_set_context_parameter(context, VERBOSE, "1");

  // Load Hypergraph for DEFAULT preset
  mt_kahypar_hypergraph_t graph =
    mt_kahypar_read_hypergraph_from_file("delaunay_n15.graph",
    DEFAULT, METIS /* file format */);

  // Partition Hypergraph
  mt_kahypar_partitioned_hypergraph_t partitioned_graph =
    mt_kahypar_partition(graph, context);

  // Extract Partition
  std::unique_ptr<mt_kahypar_partition_id_t[]> partition =
    std::make_unique<mt_kahypar_partition_id_t[]>(mt_kahypar_num_hypernodes(graph));
  mt_kahypar_get_partition(partitioned_graph, partition.get());

  // Extract Block Weights
  std::unique_ptr<mt_kahypar_hypernode_weight_t[]> block_weights =
    std::make_unique<mt_kahypar_hypernode_weight_t[]>(2);
  mt_kahypar_get_block_weights(partitioned_graph, block_weights.get());

  // Compute Metrics
  const double imbalance = mt_kahypar_imbalance(partitioned_graph, context);
  const double cut = mt_kahypar_cut(partitioned_graph);

  // Output Results
  std::cout << "Partitioning Results:" << std::endl;
  std::cout << "Imbalance         = " << imbalance << std::endl;
  std::cout << "Cut               = " << cut << std::endl;
  std::cout << "Weight of Block 0 = " << block_weights[0] << std::endl;
  std::cout << "Weight of Block 1 = " << block_weights[1] << std::endl;

  mt_kahypar_free_context(context);
  mt_kahypar_free_hypergraph(graph);
  mt_kahypar_free_partitioned_hypergraph(partitioned_graph);
}