#include <memory>
#include <vector>
#include <iostream>
#include <thread>

#include <libmtkahypar.h>

// Install library interface via 'sudo make install.mtkahypar' in build folder
// Compile with: g++ -std=c++14 -DNDEBUG -O3 partition_with_fixed_vertices.cc -o example -lmtkahypar
int main(int argc, char* argv[]) {

  // Initialize thread pool
  mt_kahypar_initialize_thread_pool(
    std::thread::hardware_concurrency() /* use all available cores */,
    true /* activate interleaved NUMA allocation policy */ );

  // Setup partitioning context
  mt_kahypar_context_t* context = mt_kahypar_context_new();
  mt_kahypar_load_preset(context, DEFAULT /* corresponds to MT-KaHyPar-D */);
  // In the following, we partition a hypergraph into four blocks
  // with an allowed imbalance of 3% and optimize the connective metric (KM1)
  mt_kahypar_set_partitioning_parameters(context,
    4 /* number of blocks */, 0.03 /* imbalance parameter */,
    KM1 /* objective function */);
  mt_kahypar_set_seed(42 /* seed */);
  // Enable logging
  mt_kahypar_set_context_parameter(context, VERBOSE, "1");

  // Load Hypergraph for DEFAULT preset
  mt_kahypar_hypergraph_t hypergraph =
    mt_kahypar_read_hypergraph_from_file("ibm01.hgr",
      DEFAULT, HMETIS /* file format */);

  // Add fixed vertices from a fixed vertex file
  std::unique_ptr<mt_kahypar_partition_id_t[]> fixed_vertices =
    std::make_unique<mt_kahypar_partition_id_t[]>(mt_kahypar_num_hypernodes(hypergraph));
  for ( size_t i = 0; i < 100; ++i ) {
    std::cout << fixed_vertices[i] << std::endl;
  }
  mt_kahypar_read_fixed_vertices_from_file("ibm01.k4.p1.fix", fixed_vertices.get());
  mt_kahypar_add_fixed_vertices(hypergraph, fixed_vertices.get(), 4 /* number of blocks */);
  // Or simply add the fixed vertices of the file directly to the hypergraph:
  // mt_kahypar_add_fixed_vertices_from_file(
  //   hypergraph, "ibm01.k4.p1.fix", 4 /* number of blocks */);

  // Partition Hypergraph
  mt_kahypar_partitioned_hypergraph_t partitioned_hg =
    mt_kahypar_partition(hypergraph, context);

  // Extract Partition
  std::unique_ptr<mt_kahypar_partition_id_t[]> partition =
    std::make_unique<mt_kahypar_partition_id_t[]>(mt_kahypar_num_hypernodes(hypergraph));
  mt_kahypar_get_partition(partitioned_hg, partition.get());

  // Extract Block Weights
  std::unique_ptr<mt_kahypar_hypernode_weight_t[]> block_weights =
    std::make_unique<mt_kahypar_hypernode_weight_t[]>(2);
  mt_kahypar_get_block_weights(partitioned_hg, block_weights.get());

  // Compute Metrics
  const double imbalance = mt_kahypar_imbalance(partitioned_hg, context);
  const double km1 = mt_kahypar_km1(partitioned_hg);

  // Output Results
  std::cout << "Partitioning Results:" << std::endl;
  std::cout << "Imbalance         = " << imbalance << std::endl;
  std::cout << "Km1               = " << km1 << std::endl;

  bool correct_assignment = true;
  for ( mt_kahypar_hypernode_id_t hn = 0; hn < mt_kahypar_num_hypernodes(hypergraph); ++hn ) {
    if ( fixed_vertices[hn] != -1 && fixed_vertices[hn] != partition[hn] ) {
      std::cout << "Node " << hn << " is fixed to block " << fixed_vertices[hn]
                << ", but assigned to block " << partition[hn] << std::endl;
      correct_assignment = false;
    }
  }

  if ( correct_assignment ) {
    std::cout << "\033[1;92mFixed vertex assignment was successful :)\033[0m" << std::endl;
  }

  mt_kahypar_free_context(context);
  mt_kahypar_free_hypergraph(hypergraph);
  mt_kahypar_free_partitioned_hypergraph(partitioned_hg);
}