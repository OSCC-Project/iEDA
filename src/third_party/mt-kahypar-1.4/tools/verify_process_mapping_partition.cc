/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2020 Tobias Heuer <tobias.heuer@kit.edu>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <functional>

#include "mt-kahypar/macros.h"
#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/metrics.h"
#include "mt-kahypar/datastructures/static_graph.h"
#include "mt-kahypar/datastructures/static_hypergraph.h"
#include "mt-kahypar/datastructures/partitioned_hypergraph.h"
#include "mt-kahypar/io/hypergraph_io.h"
#include "mt-kahypar/io/hypergraph_factory.h"
#include "mt-kahypar/io/partitioning_output.h"
#include "mt-kahypar/partition/mapping/target_graph.h"
#include "mt-kahypar/partition/mapping/initial_mapping.h"
#include "mt-kahypar/utils/timer.h"
#include "mt-kahypar/utils/randomize.h"

using namespace mt_kahypar;
namespace po = boost::program_options;
using Graph = ds::StaticGraph;
using Hypergraph = ds::StaticHypergraph;
using PartitionedHypergraph = ds::PartitionedHypergraph<Hypergraph, ds::ConnectivityInfo>;

int main(int argc, char* argv[]) {
  Context context;
  po::options_description options("Options");
  options.add_options()
    ("hypergraph,h",
     po::value<std::string>(&context.partition.graph_filename)->value_name("<string>")->required(),
     "Hypergraph Filename")
    ("partition-file,b",
     po::value<std::string>(&context.partition.graph_partition_filename)->value_name("<string>")->required(),
     "Partition Filename")
    ("process-graph-file,p",
     po::value<std::string>(&context.mapping.target_graph_file)->value_name("<string>")->required(),
     "Target Graph Filename")
    ("input-file-format",
      po::value<std::string>()->value_name("<string>")->notifier([&](const std::string& s) {
        if (s == "hmetis") {
          context.partition.file_format = FileFormat::hMetis;
        } else if (s == "metis") {
          context.partition.file_format = FileFormat::Metis;
        }
      }),
      "Input file format: \n"
      " - hmetis : hMETIS hypergraph file format \n"
      " - metis : METIS graph file format")
    ("verbose,v",
     po::value<bool>(&context.partition.verbose_output)->value_name("<bool>")->default_value(false),
     "Enables logging");

  po::variables_map cmd_vm;
  po::store(po::parse_command_line(argc, argv, options), cmd_vm);
  po::notify(cmd_vm);

  // Setup context
  context.partition.objective = Objective::steiner_tree;
  context.partition.epsilon = 0.03;
  context.shared_memory.num_threads = std::thread::hardware_concurrency();
  context.mapping.max_steiner_tree_size = 4;
  TBBInitializer::instance(context.shared_memory.num_threads);

  // Read Hypergraph
  Hypergraph hg = io::readInputFile<Hypergraph>(
    context.partition.graph_filename, context.partition.file_format, true, true);

  // Read Target Graph
  TargetGraph target_graph(io::readInputFile<Graph>(
    context.mapping.target_graph_file, FileFormat::Metis, true, true));
  context.partition.k = target_graph.numBlocks();
  context.setupPartWeights(hg.totalWeight());

  // Read Partition
  std::vector<PartitionID> partition;
  io::readPartitionFile(context.partition.graph_partition_filename, partition);
  PartitionedHypergraph partitioned_hg(context.partition.k, hg, parallel_tag_t { });
  partitioned_hg.doParallelForAllNodes([&](const HypernodeID& hn) {
    partitioned_hg.setOnlyNodePart(hn, partition[hn]);
  });
  partitioned_hg.initializePartition();
  partitioned_hg.setTargetGraph(&target_graph);

  // Precompute Steiner Trees
  HighResClockTimepoint start = std::chrono::high_resolution_clock::now();
  target_graph.precomputeDistances(
    std::min(context.mapping.max_steiner_tree_size,
      static_cast<size_t>(hg.maxEdgeSize())));
  HighResClockTimepoint end = std::chrono::high_resolution_clock::now();

  if ( context.partition.verbose_output ) {
    std::chrono::duration<double> elapsed_seconds(end - start);
    io::printPartitioningResults(partitioned_hg, context, elapsed_seconds);
  }

  std::cout << "RESULT"
            << " graph=" << context.partition.graph_filename.substr(
                context.partition.graph_filename.find_last_of('/') + 1)
            << " partition_file=" << context.partition.graph_partition_filename.substr(
                context.partition.graph_partition_filename.find_last_of('/') + 1)
            << " target_graph_file=" << context.mapping.target_graph_file.substr(
               context.mapping.target_graph_file.find_last_of('/') + 1)
            << " objective=" << context.partition.objective
            << " k=" << context.partition.k
            << " epsilon=" << context.partition.epsilon
            << " imbalance=" << metrics::imbalance(partitioned_hg, context)
            << " steiner_tree=" << metrics::quality(partitioned_hg, Objective::steiner_tree)
            << " approximation_factor=" << metrics::approximationFactorForProcessMapping(partitioned_hg, context)
            << " cut=" << metrics::quality(partitioned_hg, Objective::cut)
            << " km1=" << metrics::quality(partitioned_hg, Objective::km1)
            << " soed=" << metrics::quality(partitioned_hg, Objective::soed)
            << std::endl;

  return 0;
}
