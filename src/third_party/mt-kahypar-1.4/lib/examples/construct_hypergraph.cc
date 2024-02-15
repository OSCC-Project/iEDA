#include <memory>
#include <vector>
#include <iostream>
#include <thread>

#include <libmtkahypar.h>

// Install library interface via 'sudo make install.mtkahypar' in build folder
// Compile with: g++ -std=c++14 -DNDEBUG -O3 construct_graph.cc -o example -lmtkahypar
int main(int argc, char* argv[]) {

  // Initialize thread pool
  mt_kahypar_initialize_thread_pool(
    std::thread::hardware_concurrency() /* use all available cores */,
    true /* activate interleaved NUMA allocation policy */ );

  // In the following, we construct a hypergraph with 7 nodes and 4 hyperedges
  const mt_kahypar_hypernode_id_t num_nodes = 7;
  const mt_kahypar_hyperedge_id_t num_hyperedges = 4;

  // The hyperedge indices points to the hyperedge vector and defines the
  // the ranges containing the pins of each hyperedge
  std::unique_ptr<size_t[]> hyperedge_indices = std::make_unique<size_t[]>(5);
  hyperedge_indices[0] = 0; hyperedge_indices[1] = 2; hyperedge_indices[2] = 6;
  hyperedge_indices[3] = 9; hyperedge_indices[4] = 12;

  std::unique_ptr<mt_kahypar_hyperedge_id_t[]> hyperedges =
    std::make_unique<mt_kahypar_hyperedge_id_t[]>(12);
  // First hyperedge contains nodes with ID 0 and 2
  hyperedges[0] = 0;  hyperedges[1] = 2;
  // Second hyperedge contains nodes with ID 0, 1, 3 and 4
  hyperedges[2] = 0;  hyperedges[3] = 1; hyperedges[4] = 3;  hyperedges[5] = 4;
  // Third hyperedge contains nodes with ID 3, 4 and 6
  hyperedges[6] = 3;  hyperedges[7] = 4; hyperedges[8] = 6;
  // Fourth hyperedge contains nodes with ID 2, 5 and 6
  hyperedges[9] = 2; hyperedges[10] = 5; hyperedges[11] = 6;

  // Define node weights
  std::unique_ptr<mt_kahypar_hypernode_weight_t[]> node_weights =
    std::make_unique<mt_kahypar_hypernode_weight_t[]>(7);
  node_weights[0] = 2; node_weights[1] = 1; node_weights[2] = 2; node_weights[3] = 4;
  node_weights[4] = 1; node_weights[5] = 3; node_weights[6] = 3;

  // Define hyperedge weights
  std::unique_ptr<mt_kahypar_hyperedge_weight_t[]> hyperedge_weights =
    std::make_unique<mt_kahypar_hyperedge_weight_t[]>(4);
  hyperedge_weights[0] = 1; hyperedge_weights[1] = 10;
  hyperedge_weights[2] = 1; hyperedge_weights[3] = 10;

  // Construct hypergraph for DEFAULT preset
  mt_kahypar_hypergraph_t hypergraph =
    mt_kahypar_create_hypergraph(DEFAULT, num_nodes, num_hyperedges,
      hyperedge_indices.get(), hyperedges.get(),
      hyperedge_weights.get(), node_weights.get());

  std::cout << "Number of Nodes            = " << mt_kahypar_num_hypernodes(hypergraph) << std::endl;
  std::cout << "Number of Hyperedges       = " << mt_kahypar_num_hyperedges(hypergraph) << std::endl;
  std::cout << "Number of Pins             = " << mt_kahypar_num_pins(hypergraph) << std::endl;
  std::cout << "Total Weight of Hypergraph = " << mt_kahypar_hypergraph_weight(hypergraph) << std::endl;

  mt_kahypar_free_hypergraph(hypergraph);
}