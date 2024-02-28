#include "MtKahypar.hh"

#include <libmtkahypar.h>
#include <libmtkahypartypes.h>

#include <iostream>
#include <memory>

namespace imp {
std::vector<size_t> MtKahypar::operator()(const std::string name, const std::vector<size_t>& eptr, const std::vector<size_t>& eind,
                                          size_t nparts, const std::vector<int>& vwgts, const std::vector<int>& hewgts)
{
  mt_kahypar_initialize_thread_pool(num_threads, true /* activate interleaved NUMA allocation policy */);

  // Setup partitioning context
  mt_kahypar_context_t* context = mt_kahypar_context_new();
  mt_kahypar_load_preset(context, DEFAULT /* corresponds to MT-KaHyPar-D */);
  // In the following, we partition a hypergraph into two blocks
  // with an allowed imbalance of 3% and optimize the connective metric (KM1)
  mt_kahypar_set_partitioning_parameters(context, nparts /* number of blocks */, 0.03 /* imbalance parameter */,
                                         KM1 /* objective function */);
  mt_kahypar_set_seed(42 /* seed */);
  // Enable logging
  mt_kahypar_set_context_parameter(context, VERBOSE, "1");
  mt_kahypar_hyperedge_id_t num_hedges = eptr.size() - 1;
  mt_kahypar_hypernode_id_t num_vertices
      = !vwgts.empty() ? vwgts.size() : *std::max_element(std::begin(eind), std::end(eind)) + mt_kahypar_hypernode_id_t(1);

  const mt_kahypar_hyperedge_id_t* eind_ptr = nullptr;
  mt_kahypar_hyperedge_id_t* eind_ptr_t = nullptr;
  constexpr bool is_same_type = std::is_same_v<mt_kahypar_hyperedge_id_t, size_t>;
  if constexpr (is_same_type)
    eind_ptr = eind.data();
  else {
    eind_ptr_t = new mt_kahypar_hyperedge_id_t[eind.size()];
    size_t i = 0;
    std::generate(eind_ptr_t, eind_ptr_t + eind.size(), [&]() { return eind[i++]; });
    eind_ptr = eind_ptr_t;
  }

  mt_kahypar_hypergraph_t hypergraph
      = mt_kahypar_create_hypergraph(DEFAULT, num_vertices, num_hedges, eptr.data(), eind_ptr, hewgts.data(), vwgts.data());

  if constexpr (!is_same_type) {
    delete[] eind_ptr_t;
    eind_ptr = nullptr;
    eind_ptr_t = nullptr;
  }

  mt_kahypar_partitioned_hypergraph_t partitioned_hg = mt_kahypar_partition(hypergraph, context);

  // Extract Partition
  std::vector<mt_kahypar_partition_id_t> partition_t(mt_kahypar_num_hypernodes(hypergraph));
  mt_kahypar_get_partition(partitioned_hg, partition_t.data());

  // Extract Block Weights
  std::vector<mt_kahypar_hypernode_weight_t> block_weights(nparts, 0);
  mt_kahypar_get_block_weights(partitioned_hg, block_weights.data());

  // Compute Metrics
  const double imbalance = mt_kahypar_imbalance(partitioned_hg, context);
  const double km1 = mt_kahypar_km1(partitioned_hg);

  // Output Results
  std::cout << "Partitioning Results:" << std::endl;
  std::cout << "Imbalance         = " << imbalance << std::endl;
  std::cout << "Km1               = " << km1 << std::endl;
  for (auto&& i : block_weights) {
    std::cout << "Weight of Block= " << i << std::endl;
  }

  mt_kahypar_free_context(context);
  mt_kahypar_free_hypergraph(hypergraph);
  mt_kahypar_free_partitioned_hypergraph(partitioned_hg);
  return {partition_t.begin(), partition_t.end()};
}

}  // namespace imp