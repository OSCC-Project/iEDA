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

bool readPartitionFile(const std::string& partition_file, PartitionedHypergraph& hypergraph) {
  bool success = true;
  std::vector<PartitionID> partition;
  mt_kahypar::io::readPartitionFile(partition_file, partition);
  for ( const HypernodeID& hn : hypergraph.nodes() ) {
    if ( partition[hn] == kInvalidPartition ) {
      LOG << RED << "[ERROR]" << END << "Hypernode" << hn << "is not assigned to a block";
      success = false;
    } else if ( partition[hn] >= hypergraph.k() ) {
      LOG << RED << "[ERROR]" << END << "Hypernode" << hn << "is assigned to block"
          << ( partition[hn] + 1 ) << ", but there are only" << hypergraph.k() << "blocks";
      success = false;
    }
    hypergraph.setOnlyNodePart(hn, partition[hn]);
  }
  hypergraph.initializePartition();
  return success;
}

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
          ("blocks,k",
           po::value<PartitionID>(&context.partition.k)->value_name("<int>")->required(),
           "Number of Blocks")
           ("epsilon,e",
           po::value<double>(&context.partition.epsilon)->value_name("<double>")->required(),
           "Imbalance")
           ("fixed-vertices,f",
           po::value<std::string>(&context.partition.fixed_vertex_filename)->value_name("<string>"),
           "Fixed Vertex File");

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

  // Add fixed vertices
  if ( context.partition.fixed_vertex_filename != "" ) {
    mt_kahypar::io::addFixedVerticesFromFile(
      hypergraph, context.partition.fixed_vertex_filename, context.partition.k);
  }

  // Setup Context
  context.setupPartWeights(hg.totalWeight());

  // Read Partition File
  bool success = readPartitionFile(context.partition.graph_partition_filename, phg);

  for ( PartitionID i = 0; i < context.partition.k; ++i ) {
    if ( phg.partWeight(i) == 0 ) {
      LOG << RED << "[ERROR]" << END << "Block" << (i + 1) << "is empty" << END;
      success = false;
    } else if ( phg.partWeight(i) > context.partition.max_part_weights[i] ) {
      LOG << RED << "[ERROR]" << END << "Block" << (i + 1) << "has weight"
          << phg.partWeight(i) << ", but maximum allowed block weight is"
          << context.partition.max_part_weights[i] << END;
      success = false;
    }
  }

  // Check fixed vertices
  if ( phg.hasFixedVertices() ) {
    for ( const HypernodeID& hn : phg.nodes() ) {
      if ( phg.isFixed(hn) && phg.fixedVertexBlock(hn) != phg.partID(hn) ) {
        LOG << RED << "Node" << hn << "is fixed to block" << phg.fixedVertexBlock(hn)
            << ", but assigned to block" << phg.partID(hn) << END;
        success = false;
      }
    }
  }

  utils::delete_hypergraph(hypergraph);

  return success ? 0 : -1;
}
