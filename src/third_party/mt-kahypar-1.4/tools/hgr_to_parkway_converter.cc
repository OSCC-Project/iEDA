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
#include <string>

#include "mt-kahypar/macros.h"
#include "mt-kahypar/datastructures/static_hypergraph.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/io/hypergraph_factory.h"
#include "mt-kahypar/io/hypergraph_io.h"
#include "mt-kahypar/utils/cast.h"
#include "mt-kahypar/utils/delete.h"

using namespace mt_kahypar;
namespace po = boost::program_options;

using HypernodeID = mt_kahypar::HypernodeID;
using HyperedgeID = mt_kahypar::HyperedgeID;
using Hypergraph = ds::StaticHypergraph;

static void writeParkwayHypergraphForProc(const Hypergraph& hypergraph,
                                          const std::string& hgr_filename,
                                          const size_t num_procs,
                                          const int rank) {
  const size_t num_vertices = hypergraph.initialNumNodes();
  const size_t num_edges = hypergraph.initialNumEdges();
  const size_t vertices_per_proc = num_vertices / num_procs;
  const size_t hyperedges_per_proc = num_edges / num_procs;

  const HypernodeID hn_start = rank * vertices_per_proc;
  const HypernodeID hn_end = static_cast<size_t>(rank) != num_procs - 1 ?
    (rank + 1) * vertices_per_proc : num_vertices;
  const HyperedgeID he_start = rank * hyperedges_per_proc;
  const HyperedgeID he_end = static_cast<size_t>(rank) != num_procs - 1 ?
    (rank + 1) * hyperedges_per_proc : num_edges;

  std::vector<int> hypernode_weight;
  for ( HypernodeID hn = hn_start; hn < hn_end; ++hn ) {
    hypernode_weight.push_back(hypergraph.nodeWeight(hn));
  }

  std::vector<int> hyperedge_data;
  for ( HyperedgeID he = he_start; he < he_end; ++he ) {
    hyperedge_data.push_back(static_cast<int>(hypergraph.edgeSize(he)) + 2);
    hyperedge_data.push_back(static_cast<int>(hypergraph.edgeWeight(he)));
    for ( const HypernodeID& pin : hypergraph.pins(he) ) {
      hyperedge_data.push_back(static_cast<int>(pin));
    }
  }

  int num_hypernodes = static_cast<int>(num_vertices);
  int num_local_hypernodes = static_cast<int>(hn_end - hn_start);
  int hyperedge_data_length = static_cast<int>(hyperedge_data.size());

  std::string out_file = hgr_filename + "-" + std::to_string(rank);
  std::ofstream out_stream(out_file, std::ofstream::out | std::ofstream::binary);

  out_stream.write((char *) &num_hypernodes, sizeof(int));
  out_stream.write((char *) &num_local_hypernodes, sizeof(int));
  out_stream.write((char *) &hyperedge_data_length, sizeof(int));
  out_stream.write((char *) hypernode_weight.data(), sizeof(int) * num_local_hypernodes);
  out_stream.write((char *) hyperedge_data.data(), sizeof(int) * hyperedge_data_length);
  out_stream.close();
}

int main(int argc, char* argv[]) {
  std::string hgr_filename;
  std::string out_filename;
  int num_procs;

  po::options_description options("Options");
  options.add_options()
    ("hypergraph,h",
    po::value<std::string>(&hgr_filename)->value_name("<string>")->required(),
    "Hypergraph filename")
    ("num-procs,p",
    po::value<int>(&num_procs)->value_name("<int>")->required(),
    "Number of Processor Parkway will be called with")
    ("out-file,o",
    po::value<std::string>(&out_filename)->value_name("<string>")->required(),
    "Hypergraph Output Filename");

  po::variables_map cmd_vm;
  po::store(po::parse_command_line(argc, argv, options), cmd_vm);
  po::notify(cmd_vm);

  // Read Hypergraph
  mt_kahypar_hypergraph_t hypergraph =
    mt_kahypar::io::readInputFile(
      hgr_filename, PresetType::default_preset,
      InstanceType::hypergraph, FileFormat::hMetis, true);
  Hypergraph& hg = utils::cast<Hypergraph>(hypergraph);

  for ( int p = 0; p < num_procs; ++p ) {
    writeParkwayHypergraphForProc(hg, out_filename, num_procs, p);
  }

  utils::delete_hypergraph(hypergraph);

  return 0;
}
