// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file PwrDumpSeqGraph.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief Dump the seq graph for debug.
 * @version 0.1
 * @date 2023-03-06
 */

#include "PwrDumpSeqGraph.hh"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <regex>
#include <string>

#include "string/Str.hh"

namespace ipower {

unsigned PwrDumpSeqYaml::operator()(PwrSeqVertex* the_vertex) {
  YAML::Node& node = _node;

  static int node_id = 0;
  std::string node_name = Str::printf("node_%d", node_id++);
  YAML::Node vertex_node;
  node[node_name] = vertex_node;
  std::string obj_name(the_vertex->get_obj_name().data(),
                       the_vertex->get_obj_name().length());
  vertex_node["name"] = obj_name;

  YAML::Node node1;
  vertex_node["attribute"] = node1;
  node1["is_input_port"] = the_vertex->isInputPort();
  node1["is_output_port"] = the_vertex->isOutputPort();
  node1["is_const"] = the_vertex->isConst();
  node1["is_level_set"] = the_vertex->isLevelSet();
  node1["level"] = the_vertex->get_level();

  YAML::Node node2;
  vertex_node["src_arcs"] = node2;
  for (auto* src_arc : the_vertex->get_src_arcs()) {
    std::string snk_vertex_name(src_arc->get_snk()->get_obj_name().data(),
                                src_arc->get_snk()->get_obj_name().length());
    node2.push_back(snk_vertex_name);
  }
  YAML::Node node3;
  vertex_node["snk_arcs"] = node3;
  for (auto* snk_arc : the_vertex->get_snk_arcs()) {
    std::string src_vertex_name(snk_arc->get_src()->get_obj_name().data(),
                                snk_arc->get_src()->get_obj_name().length());
    node3.push_back(src_vertex_name);
  }

  return 1;
}

/**
 * @brief Print the yaml to text.
 *
 * @param file_name
 */
void PwrDumpSeqYaml::printText(const char* file_name) {
  std::ofstream file(file_name);
  file << _node << std::endl;
  file.close();
}

/**
 * @brief Print Graph in GraphViz dot format.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned PwrDumpSeqGraphViz::operator()(PwrSeqGraph* the_graph) {
  LOG_INFO << "dump seq graph dotviz start";

  std::ofstream dot_file;
  dot_file.open("/home/shaozheqing/iSO/src/iPower/test/digraph.dot");

  dot_file << "digraph g {"
           << "\n";

  PwrSeqArc* seq_arc;
  FOREACH_SEQ_ARC(the_graph, seq_arc) {
    auto src_name = seq_arc->get_src()->get_own_seq_inst()->getFullName();
    auto snk_name = seq_arc->get_snk()->get_own_seq_inst()->getFullName();

    auto new_src_name = Str::replace(src_name, "[:\\[\\]]", "_");
    auto new_snk_name = Str::replace(snk_name, "[:\\[\\]]", "_");

    dot_file << new_src_name << " -> " << new_snk_name << " "
             << "\n";
  }

  dot_file << "}"
           << "\n";

  dot_file.close();

  LOG_INFO << "dump graph dotviz end";

  return 1;
}
}  // namespace ipower