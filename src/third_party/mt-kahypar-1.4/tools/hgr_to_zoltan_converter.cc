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

using Hypergraph = ds::StaticHypergraph;

static void writeZoltanHypergraph(const Hypergraph& hypergraph,
                                  const std::string& hgr_filename) {
  std::ofstream out_stream(hgr_filename.c_str());
  out_stream << 0;                     // 0-based indexing
  out_stream << " " << hypergraph.initialNumNodes() << " " << hypergraph.initialNumEdges() << " " << hypergraph.initialNumPins();
  out_stream << " " << 3 << std::endl;  // weighting scheme: both edge and node weights

  for (const HyperedgeID& he : hypergraph.edges()) {
    out_stream << hypergraph.edgeWeight(he) << " ";
    for (const HypernodeID& pin : hypergraph.pins(he)) {
      out_stream << pin << " ";
    }
    out_stream << "\n";
  }

  for (const HypernodeID& hn : hypergraph.nodes()) {
    out_stream << hypergraph.nodeWeight(hn) << "\n";
  }
  out_stream << std::endl;
  out_stream.close();
}

int main(int argc, char* argv[]) {
  std::string hgr_filename;
  std::string out_filename;

  po::options_description options("Options");
  options.add_options()
    ("hypergraph,h",
    po::value<std::string>(&hgr_filename)->value_name("<string>")->required(),
    "Hypergraph filename")
    ("out-file,o",
    po::value<std::string>(&out_filename)->value_name("<string>")->required(),
    "Hypergraph Output Filename");

  po::variables_map cmd_vm;
  po::store(po::parse_command_line(argc, argv, options), cmd_vm);
  po::notify(cmd_vm);

  mt_kahypar_hypergraph_t hypergraph =
    mt_kahypar::io::readInputFile(
      hgr_filename, PresetType::default_preset,
      InstanceType::hypergraph, FileFormat::hMetis, true);
  Hypergraph& hg = utils::cast<Hypergraph>(hypergraph);

  writeZoltanHypergraph(hg, out_filename);

  utils::delete_hypergraph(hypergraph);

  return 0;
}
