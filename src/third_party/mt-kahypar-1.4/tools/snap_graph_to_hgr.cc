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

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  std::string graph_filename;
  std::string hgr_filename;
  int num_nodes;
  int num_edges;

  po::options_description options("Options");
  options.add_options()
    ("graph,g",
    po::value<std::string>(&graph_filename)->value_name("<string>")->required(),
    "Graph filename")
    ("hypergraph,h",
    po::value<std::string>(&hgr_filename)->value_name("<string>")->required(),
    "Hypergraph filename")
    ("num-nodes",
    po::value<int>(&num_nodes)->value_name("<int>")->required(),
    "Number of Nodes")
    ("num-edges",
    po::value<int>(&num_edges)->value_name("<int>")->required(),
    "Number of Edges");

  po::variables_map cmd_vm;
  po::store(po::parse_command_line(argc, argv, options), cmd_vm);
  po::notify(cmd_vm);

  std::ofstream out_stream(hgr_filename.c_str());

  std::ifstream in_stream(graph_filename);
  std::string line;
  std::getline(in_stream, line);

  // skip any comment
  while (line[0] == '#') {
    std::getline(in_stream, line);
  }

  // Read graph edges
  std::vector<std::pair<int, int> > edges;
  int max_node_id = 0;
  for (int i = 0; i < num_edges; ++i) {
    std::istringstream sstream(line);
    int u, v;
    sstream >> u >> v;
    max_node_id = std::max(max_node_id, std::max(u, v));
    edges.emplace_back(std::make_pair(u, v));
    std::getline(in_stream, line);
  }

  int node_id = 0;
  std::vector<int> node_mapping(max_node_id + 1, -1);
  for (const auto& edge : edges) {
    int u = edge.first;
    int v = edge.second;
    if (node_mapping[u] == -1) {
      node_mapping[u] = ++node_id;
    }
    if (node_mapping[v] == -1) {
      node_mapping[v] = ++node_id;
    }
  }

  // Write header
  out_stream << num_edges << " " << num_nodes << " 0"  /* Unweighted */ << std::endl;

  // Write hyperedges
  for (const auto& edge : edges) {
    int u = edge.first;
    int v = edge.second;
    if (u != v) {
      out_stream << node_mapping[u] << " " << node_mapping[v] << std::endl;
    } else {
      out_stream << node_mapping[u] << std::endl;
    }
  }

  in_stream.close();
  out_stream.close();

  return 0;
}
