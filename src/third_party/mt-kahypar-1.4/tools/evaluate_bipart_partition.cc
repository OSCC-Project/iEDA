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
#include <sstream>
#include <string>

#include "mt-kahypar/macros.h"
#include "mt-kahypar/datastructures/static_hypergraph.h"
#include "mt-kahypar/datastructures/partitioned_hypergraph.h"
#include "mt-kahypar/datastructures/connectivity_info.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/metrics.h"
#include "mt-kahypar/io/hypergraph_factory.h"
#include "mt-kahypar/io/hypergraph_io.h"
#include "mt-kahypar/utils/cast.h"
#include "mt-kahypar/utils/delete.h"


using namespace mt_kahypar;
namespace po = boost::program_options;

using Hypergraph = ds::StaticHypergraph;
using PartitionedHypergraph = ds::PartitionedHypergraph<Hypergraph, ds::ConnectivityInfo>;

void readBipartPartitionFile(const std::string& bipart_partition_file,
                             PartitionedHypergraph& hypergraph,
                             const PartitionID k) {
  ASSERT(!bipart_partition_file.empty(), "No filename for partition file specified");
  std::ifstream file(bipart_partition_file);
  if (file) {
    for ( PartitionID block = 0; block < k; ++block ) {
      std::string line;
      std::getline(file, line);
      std::istringstream line_stream(line);
      HypernodeID hn = 0;
      PartitionID bipart_block = 0;
      line_stream >> bipart_block;
      ASSERT(block == bipart_block - 1);
      while ( line_stream >> hn ) {
        hypergraph.setOnlyNodePart(hn - 1, block);
      }
    }
    hypergraph.initializePartition();
    file.close();
  } else {
    std::cerr << "Error: File not found: " << std::endl;
  }
}

int main(int argc, char* argv[]) {
  Context context;

  po::options_description options("Options");
  options.add_options()
          ("hypergraph,h",
           po::value<std::string>(&context.partition.graph_filename)->value_name("<string>")->required(),
           "Hypergraph Filename")
          ("bipart-partition-file,b",
           po::value<std::string>(&context.partition.graph_partition_filename)->value_name("<string>")->required(),
           "BiPart Partition Filename")
          ("blocks,k",
           po::value<PartitionID>(&context.partition.k)->value_name("<int>")->required(),
           "Number of Blocks");

  po::variables_map cmd_vm;
  po::store(po::parse_command_line(argc, argv, options), cmd_vm);
  po::notify(cmd_vm);

  // Read Hypergraph
  mt_kahypar_hypergraph_t hypergraph =
    mt_kahypar::io::readInputFile(
      context.partition.graph_filename, PresetType::default_preset,
      InstanceType::hypergraph, FileFormat::hMetis, true);
  Hypergraph& hg = utils::cast<Hypergraph>(hypergraph);
  PartitionedHypergraph phg(context.partition.k, hg, parallel_tag_t());

  // Setup Context
  context.partition.epsilon = 0.03;
  context.setupPartWeights(hg.totalWeight());

  // Read Bipart Partition File
  readBipartPartitionFile(context.partition.graph_partition_filename, phg,
                          context.partition.k);

  std::cout << "RESULT"
            << " graph=" << context.partition.graph_filename
            << " k=" << context.partition.k
            << " imbalance=" << metrics::imbalance(phg, context)
            << " cut=" << metrics::quality(phg, Objective::cut)
            << " km1=" << metrics::quality(phg, Objective::km1) << std::endl;

  utils::delete_hypergraph(hypergraph);

  return 0;
}
