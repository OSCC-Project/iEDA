#include <memory>
#include <vector>
#include <iostream>
#include <thread>

#include <libmtkahypar.h>

// Install library interface via 'sudo make install.mtkahypar' in build folder
// Compile with: g++ -std=c++14 -DNDEBUG -O3 improve_partition.cc -o example -lmtkahypar
int main(int argc, char* argv[]) {

  // Initialize thread pool
  mt_kahypar_initialize_thread_pool(
    std::thread::hardware_concurrency() /* use all available cores */,
    true /* activate interleaved NUMA allocation policy */ );

  // Setup partitioning context
  mt_kahypar_context_t* context = mt_kahypar_context_new();
  mt_kahypar_load_preset(context, DEFAULT /* corresponds to MT-KaHyPar-D */);
  // In the following, we partition a hypergraph into eight blocks
  // with an allowed imbalance of 3% and optimize the connective metric (KM1)
  mt_kahypar_set_partitioning_parameters(context,
    8 /* number of blocks */, 0.03 /* imbalance parameter */,
    KM1 /* objective function */);
  mt_kahypar_set_seed(42 /* seed */);
  // Enable logging
  mt_kahypar_set_context_parameter(context, VERBOSE, "1");

 // Load Hypergraph for DEFAULT preset
  mt_kahypar_hypergraph_t hypergraph =
    mt_kahypar_read_hypergraph_from_file("ibm01.hgr",
      DEFAULT, HMETIS /* file format */);

  // Read Partition File, which we improve with the QUALITY preset
  mt_kahypar_partitioned_hypergraph_t partitioned_hg =
    mt_kahypar_read_partition_from_file(
      hypergraph, QUALITY, 8 /* number of blocks */, "ibm01.hgr.part8");
  const double km1_before = mt_kahypar_km1(partitioned_hg);

  // Improve Partition
  mt_kahypar_load_preset(context, QUALITY /* use quality preset for improvement */);
  mt_kahypar_improve_partition(partitioned_hg, context,
    1 /* perform one multilevel improvement cycle (also called V-cycle) */);
  const double km1_after = mt_kahypar_km1(partitioned_hg);

  // Output Results
  std::cout << "Partitioning Results:" << std::endl;
  std::cout << "Km1 before Improvement Cycle = " << km1_before << std::endl;
  std::cout << "Km1 after Improvement Cycle  = " << km1_after << std::endl;

  mt_kahypar_free_context(context);
  mt_kahypar_free_hypergraph(hypergraph);
  mt_kahypar_free_partitioned_hypergraph(partitioned_hg);
}