/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2023 Tobias Heuer <tobias.heuer@kit.edu>
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
#include <functional>

#include "mt-kahypar/macros.h"
#include "mt-kahypar/datastructures/hypergraph_common.h"
#include "mt-kahypar/utils/randomize.h"

using namespace mt_kahypar;
namespace po = boost::program_options;

using AdjList = std::vector<std::vector<std::pair<HypernodeID, HyperedgeWeight>>>;

void construct_hierarchical_process_graph(AdjList& graph,
                                          const HypernodeID core_start,
                                          const HypernodeID core_end,
                                          const std::vector<HypernodeID>& hierarchy,
                                          const std::vector<HyperedgeWeight>& costs,
                                          const size_t cur_level) {
  if ( cur_level >= hierarchy.size() ) return;

  const HypernodeID num_nodes = core_end - core_start;
  const HypernodeID a = hierarchy[cur_level];
  const HypernodeID cores_per_node = num_nodes / a;
  const HyperedgeWeight c = costs[cur_level];
  for ( HypernodeID core_1 = core_start; core_1 < core_end; ++core_1 ) {
    for ( HypernodeID core_2 = core_1 + 1; core_2 < core_end; ++core_2 ) {
      if ( core_1 / cores_per_node != core_2 / cores_per_node ) {
        graph[core_1].push_back(std::make_pair(core_2, c));
        graph[core_2].push_back(std::make_pair(core_1, c));
      }
    }
  }

  for ( HypernodeID i = 0; i < a; ++i ) {
    const HypernodeID next_core_start = core_start + i * cores_per_node;
    const HypernodeID next_core_end = core_start + ( i + 1 ) * cores_per_node;
    construct_hierarchical_process_graph(graph,
      next_core_start, next_core_end, hierarchy, costs, cur_level + 1);
  }
}

int main(int argc, char* argv[]) {
  std::string hierarchy_str, costs_str;
  std::string out_folder, prefix;
  po::options_description options("Options");
  options.add_options()
    ("out-folder,o",
    po::value<std::string>(&out_folder)->value_name("<string>")->required(),
    "Process Graph Output Folder")
    ("filename-prefix",
    po::value<std::string>(&prefix)->value_name("<string>")->default_value(""),
    "Prefix of output filename")
    ("hierarchy",
    po::value<std::string>(&hierarchy_str)->value_name("<string>")->required(),
    "Data center hierarchy")
    ("communication-costs",
    po::value<std::string>(&costs_str)->value_name("<string>")->required(),
    "Communications costs");

  po::variables_map cmd_vm;
  po::store(po::parse_command_line(argc, argv, options), cmd_vm);
  po::notify(cmd_vm);

  for ( size_t i = 0; i < hierarchy_str.size(); ++i )  {
    hierarchy_str[i] = hierarchy_str[i] == ':' ? ' ' : hierarchy_str[i];
  }
  for ( size_t i = 0; i < costs_str.size(); ++i )  {
    costs_str[i] = costs_str[i] == ':' ? ' ' : costs_str[i];
  }

  std::string target_graph_file = out_folder + "/" + prefix;
  HypernodeID cur;
  std::vector<HypernodeID> hierarchy;
  std::stringstream hierarchy_stream(hierarchy_str);
  while ( hierarchy_stream >> cur ) {
    hierarchy.push_back(cur);
    target_graph_file += std::to_string(cur) + "x";
  }
  std::reverse(hierarchy.begin(), hierarchy.end());
  target_graph_file[target_graph_file.size() - 1] = '.';
  target_graph_file += "graph";

  std::vector<HyperedgeWeight> costs;
  std::stringstream costs_stream(costs_str);
  while ( costs_stream >> cur ) {
    costs.push_back(cur);
  }
  std::reverse(costs.begin(), costs.end());

  HypernodeID num_nodes = 1;
  for ( const HypernodeID a : hierarchy ) {
    num_nodes *= a;
  }

  AdjList graph(num_nodes);
  construct_hierarchical_process_graph(graph, 0, num_nodes, hierarchy, costs, 0);

  HyperedgeID num_edges = 0;
  for ( HypernodeID u = 0; u < num_nodes; ++u ) {
    std::sort(graph[u].begin(), graph[u].end());
    num_edges += graph[u].size();
  }
  num_edges /= 2;

  std::ofstream out(target_graph_file.c_str());
  out << num_nodes << " " << num_edges << " 1" << std::endl;
  for ( HypernodeID u = 0; u < num_nodes; ++u ) {
    for ( const auto& edge : graph[u] ) {
      out << ( edge.first + 1 ) << " " << edge.second << " ";
    }
    out << std::endl;
  }
  out.close();

  std::cout << "Graph has been written to '" << target_graph_file << "'" << std::endl;

  return 0;
}
