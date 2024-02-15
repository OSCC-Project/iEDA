#include <memory>
#include <vector>
#include <iostream>
#include <thread>

#include <libmtkahypar.h>

// Install library interface via 'sudo make install.mtkahypar' in build folder
// Compile with: g++ -std=c++14 -DNDEBUG -O3 partition_with_individual_block_weights.cc -o example -lmtkahypar
int main(int argc, char* argv[]) {

  // Initialize thread pool
  mt_kahypar_initialize_thread_pool(
    std::thread::hardware_concurrency() /* use all available cores */,
    true /* activate interleaved NUMA allocation policy */ );

  // Setup partitioning context
  mt_kahypar_context_t* context = mt_kahypar_context_new();
  mt_kahypar_load_preset(context, DEFAULT /* corresponds to MT-KaHyPar-D */);
  // In the following, we partition a hypergraph into four blocks
  // and optimize the connective metric (KM1)
  mt_kahypar_set_partitioning_parameters(context,
    4 /* number of blocks */,
    0.03 /* imbalance parameter not relevant for partitioning with individual block weights */,
    KM1 /* objective function */);
  mt_kahypar_set_seed(42 /* seed */);
  // Enable logging
  mt_kahypar_set_context_parameter(context, VERBOSE, "1");

  // Setup Individual Block Weights
  std::unique_ptr<mt_kahypar_hypernode_weight_t[]> individual_block_weights =
    std::make_unique<mt_kahypar_hypernode_weight_t[]>(4);
  // The weight of the first block must be smaller or equal than 2131
  individual_block_weights[0] = 2131;
  // The weight of the second block must be smaller or equal than 1213
  individual_block_weights[1] = 1213;
  // The weight of the third block must be smaller or equal than 7287
  individual_block_weights[2] = 7287;
  // The weight of the fourth block must be smaller or equal than 2501
  individual_block_weights[3] = 2501;
  mt_kahypar_set_individual_target_block_weights(context, 4, individual_block_weights.get());

  // Load Hypergraph for DEFAULT preset
  mt_kahypar_hypergraph_t hypergraph =
    mt_kahypar_read_hypergraph_from_file("ibm01.hgr",
      DEFAULT, HMETIS /* file format */);

  // Partition Hypergraph
  mt_kahypar_partitioned_hypergraph_t partitioned_hg =
    mt_kahypar_partition(hypergraph, context);

  // Extract Block Weights
  std::unique_ptr<mt_kahypar_hypernode_weight_t[]> block_weights =
    std::make_unique<mt_kahypar_hypernode_weight_t[]>(2);
  mt_kahypar_get_block_weights(partitioned_hg, block_weights.get());

  // Output Results
  const double km1 = mt_kahypar_km1(partitioned_hg);
  std::cout << "Partitioning Results:" << std::endl;
  std::cout << "Km1               = " << km1 << std::endl;
  std::cout << "Weight of Block 0 = " << block_weights[0] << " (<= " << individual_block_weights[0] << ")" << std::endl;
  std::cout << "Weight of Block 1 = " << block_weights[1] << " (<= " << individual_block_weights[1] << ")" << std::endl;
  std::cout << "Weight of Block 2 = " << block_weights[2] << " (<= " << individual_block_weights[2] << ")" << std::endl;
  std::cout << "Weight of Block 3 = " << block_weights[3] << " (<= " << individual_block_weights[3] << ")" << std::endl;

  mt_kahypar_free_context(context);
  mt_kahypar_free_hypergraph(hypergraph);
  mt_kahypar_free_partitioned_hypergraph(partitioned_hg);
}