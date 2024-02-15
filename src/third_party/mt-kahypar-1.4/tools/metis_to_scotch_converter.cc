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
#include "mt-kahypar/datastructures/static_graph.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/io/hypergraph_factory.h"
#include "mt-kahypar/io/hypergraph_io.h"
#include "mt-kahypar/utils/cast.h"
#include "mt-kahypar/utils/delete.h"

using namespace mt_kahypar;
namespace po = boost::program_options;

using HypernodeID = mt_kahypar::HypernodeID;
using HyperedgeID = mt_kahypar::HyperedgeID;
using Graph = ds::StaticGraph;

static void writeScotchGraphFile(const Graph& graph,
                                 const std::string& hgr_filename) {
  std::ofstream out(hgr_filename.c_str());
  out << "0" << std::endl;
  out << graph.initialNumNodes() << " " << ( 2 * graph.initialNumEdges() ) << std::endl;
  out << "0 000" << std::endl; // we only support conversion of unweighted instances here

  for ( const HypernodeID& u : graph.nodes() ) {
    out << graph.nodeDegree(u);
    for ( const HyperedgeID& e : graph.incidentEdges(u) ) {
      for ( const HypernodeID& v : graph.pins(e) ) {
        if ( u != v ) {
          out << " " << v;
        }
      }
    }
    out << std::endl;
  }

  out.close();
}

int main(int argc, char* argv[]) {
  std::string graph_filename;
  std::string out_filename;

  po::options_description options("Options");
  options.add_options()
    ("graph,g",
    po::value<std::string>(&graph_filename)->value_name("<string>")->required(),
    "Metis filename")
    ("out-file,o",
    po::value<std::string>(&out_filename)->value_name("<string>")->required(),
    "Graph Output Filename");

  po::variables_map cmd_vm;
  po::store(po::parse_command_line(argc, argv, options), cmd_vm);
  po::notify(cmd_vm);

  // Read Hypergraph
  mt_kahypar_hypergraph_t gr =
    mt_kahypar::io::readInputFile(
      graph_filename, PresetType::default_preset,
      InstanceType::graph, FileFormat::Metis, true);
  Graph& graph = utils::cast<Graph>(gr);

  writeScotchGraphFile(graph, out_filename);

  utils::delete_hypergraph(gr);

  return 0;
}
