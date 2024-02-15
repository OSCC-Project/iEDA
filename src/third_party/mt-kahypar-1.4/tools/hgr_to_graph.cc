/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2023 Nikolai Maas <nikolai.maas@kit.edu>
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
#include <algorithm>

#include "mt-kahypar/macros.h"
#include "mt-kahypar/definitions.h"
#include "mt-kahypar/io/hypergraph_factory.h"
#include "mt-kahypar/io/hypergraph_io.h"

using namespace mt_kahypar;
namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  std::string graph_filename;
  std::string hgr_filename;

  po::options_description options("Options");
  options.add_options()
    ("hypergraph,h",
    po::value<std::string>(&hgr_filename)->value_name("<string>")->required(),
    "Hypergraph filename")
    ("graph,g",
    po::value<std::string>(&graph_filename)->value_name("<string>")->required(),
    "Graph filename");

  po::variables_map cmd_vm;
  po::store(po::parse_command_line(argc, argv, options), cmd_vm);
  po::notify(cmd_vm);

  std::ofstream out_stream(graph_filename.c_str());

  // Read Hypergraph
  HyperedgeID num_edges = 0;
  HypernodeID num_nodes = 0;
  HyperedgeID num_removed_single_pin_hyperedges = 0;
  io::HyperedgeVector hyperedges;
  vec<HyperedgeWeight> hyperedges_weight;
  vec<HypernodeWeight> hypernodes_weight;

  io::readHypergraphFile(hgr_filename, num_edges, num_nodes, num_removed_single_pin_hyperedges,
                         hyperedges, hyperedges_weight, hypernodes_weight);
  ALWAYS_ASSERT(hyperedges.size() == num_edges);
  ALWAYS_ASSERT(num_removed_single_pin_hyperedges == 0);

  // Write header
  out_stream << num_nodes << " " << num_edges << " ";
  if (hyperedges_weight.empty() && hypernodes_weight.empty()) {
    out_stream << "0"  /* Unweighted */ << std::endl;
  } else {
    out_stream << (hypernodes_weight.empty() ? "0" : "1");
    out_stream << (hyperedges_weight.empty() ? "0" : "1") << std::endl;
  }

  // insert backward edges
  hyperedges.reserve(2 * num_edges);
  for (size_t he = 0; he < num_edges; ++he) {
    const auto& pins = hyperedges[he];
    ALWAYS_ASSERT(pins.size() == 2, "Input hypergraph is not a graph!");
    hyperedges.push_back({pins[1], pins[0]});
  }

  std::sort(hyperedges.begin(), hyperedges.end(),
            [](const auto& l, const auto& r) { return l[0] < r[0] || (l[0] == r[0] && l[1] < r[1]); });

  // Write edges
  size_t i = 0;
  bool at_start_of_line = true;
  for (size_t he = 0; he < hyperedges.size();) {
    const auto& pins = hyperedges[he];
    if (!hypernodes_weight.empty() && at_start_of_line) {
      out_stream << hypernodes_weight[i] << " ";
    }
    if (pins[0] == i) {
      out_stream << (pins[1] + 1) << " ";
      if (!hyperedges_weight.empty()) {
        out_stream << hyperedges_weight[he] << " ";
      }
      ++he;
      at_start_of_line = false;
    } else {
      out_stream << std::endl;
      ++i;
      ALWAYS_ASSERT(i < num_nodes);
      at_start_of_line = true;
    }
  }

  out_stream << std::endl;
  out_stream.close();

  return 0;
}
