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

  // In the following, we construct a graph with 5 nodes and 6 edges
  const mt_kahypar_hypernode_id_t num_nodes = 5;
  const mt_kahypar_hyperedge_id_t num_edges = 6;

  // We represent the edges of the graph as edge list vector.
  // Two consecutive node IDs in the edge list vector form
  // an undirected edge in the graph.
  std::unique_ptr<mt_kahypar_hypernode_id_t[]> edges =
    std::make_unique<mt_kahypar_hypernode_id_t[]>(12);
  // The first undirected edge connects node 0 and 1
  edges[0] = 0;  edges[1] = 1;
  // The second undirected edge connects node 0 and 2
  edges[2] = 0;  edges[3] = 2;
  // The third undirected edge connects node 1 and 2
  edges[4] = 1;  edges[5] = 2;
  // The fourth undirected edge connects node 1 and 3
  edges[6] = 1;  edges[7] = 3;
  // The fifth undirected edge connects node 2 and 3
  edges[8] = 2;  edges[9] = 3;
  // The sixth undirected edge connects node 3 and 4
  edges[10] = 3; edges[11] = 4;

  // Define node weights
  std::unique_ptr<mt_kahypar_hypernode_weight_t[]> node_weights =
    std::make_unique<mt_kahypar_hypernode_weight_t[]>(5);
  node_weights[0] = 2; node_weights[1] = 1; node_weights[2] = 2;
  node_weights[3] = 4; node_weights[4] = 1;

  // Define edge weights
  std::unique_ptr<mt_kahypar_hyperedge_weight_t[]> edge_weights =
    std::make_unique<mt_kahypar_hyperedge_weight_t[]>(6);
  edge_weights[0] = 1; edge_weights[1] = 10;
  edge_weights[2] = 1; edge_weights[3] = 10;
  edge_weights[3] = 1; edge_weights[4] = 10;

  // Construct graph
  mt_kahypar_hypergraph_t graph =
    mt_kahypar_create_graph(DEFAULT, num_nodes, num_edges,
      edges.get(), edge_weights.get(), node_weights.get());

  std::cout << "Number of Nodes       = " << mt_kahypar_num_hypernodes(graph) << std::endl;
  std::cout << "Number of Edges       = " << mt_kahypar_num_hyperedges(graph) << std::endl;
  std::cout << "Total Weight of Graph = " << mt_kahypar_hypergraph_weight(graph) << std::endl;

  mt_kahypar_free_hypergraph(graph);
}