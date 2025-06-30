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
 * @file PwrDumpGraph.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class implemention of dump the power graph data.
 * @version 0.1
 * @date 2023-04-04
 */

#include "PwrDumpGraph.hh"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <regex>
#include <string>

#include "core/PwrSeqGraph.hh"
#include "string/Str.hh"

namespace ipower {

/**
 * @brief dump the power vertex information.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned PwrDumpGraphYaml::operator()(PwrVertex* the_vertex) {
  YAML::Node& node = _node;

  static int node_id = 0;
  std::string node_name = Str::printf("node_%d", node_id++);
  YAML::Node vertex_node;
  node[node_name] = vertex_node;
  vertex_node["name"] = the_vertex->getName();

  YAML::Node node1;
  vertex_node["attribute"] = node1;
  node1["is_pin"] = the_vertex->isPin();
  node1["is_macro_pin"] = the_vertex->isMacroPin();
  node1["is_seq_pin"] = the_vertex->isSeqPin();
  node1["is_seq_clock_pin"] = the_vertex->isSeqClockPin();

  node1["is_seq_visited"] = the_vertex->is_seq_visited();
  node1["is_const"] = the_vertex->is_const();
  node1["is_const_propagated"] = the_vertex->is_const_propagated();
  node1["is_const_vdd"] = the_vertex->is_const_vdd();
  node1["is_const_gnd"] = the_vertex->is_const_gnd();
  node1["is_input_port"] = the_vertex->is_input_port();
  node1["is_output_port"] = the_vertex->is_output_port();
  node1["cell_name"] =
      the_vertex->getOwnInstance()
          ? the_vertex->getOwnInstance()->get_inst_cell()->get_cell_name()
          : "NA";

  auto* seq_vertex = the_vertex->get_own_seq_vertex();
  node1["seq_vertex_level"] =
      seq_vertex ? Str::printf("%d", seq_vertex->get_level()) : "Nil";

  if (seq_vertex) {
    {
      YAML::Node seq_fanout_node;
      node1["seq_src_arcs"] = seq_fanout_node;
      PwrSeqArc* src_seq_arc;
      FOREACH_SRC_SEQ_ARC(seq_vertex, src_seq_arc) {
        auto* snk_vertex = src_seq_arc->get_snk();
        std::string snk_seq_name = std::string(snk_vertex->get_obj_name());
        snk_seq_name += Str::printf(" level %d", snk_vertex->get_level());
        seq_fanout_node.push_back(std::string(snk_seq_name));
      }
    }

    {
      YAML::Node seq_fanin_node;
      node1["seq_snk_arcs"] = seq_fanin_node;
      PwrSeqArc* snk_seq_arc;
      FOREACH_SNK_SEQ_ARC(seq_vertex, snk_seq_arc) {
        auto* src_vertex = snk_seq_arc->get_src();
        std::string src_seq_name = std::string(src_vertex->get_obj_name());
        src_seq_name += Str::printf(" level %d", src_vertex->get_level());
        seq_fanin_node.push_back(src_seq_name);
      }
    }
  }

  // print port function.
  auto* design_obj = the_vertex->get_sta_vertex()->get_design_obj();
  std::string lib_port_func_str;
  if (design_obj->isPin() && design_obj->isOutput()) {
    lib_port_func_str =
        dynamic_cast<Pin*>(design_obj)->get_cell_port()->get_func_expr_str();
  }
  node1["port_func"] = !lib_port_func_str.empty() ? lib_port_func_str : "Nil";

  auto print_toggle = [](auto& node, auto* the_vertex) {
    // dump power vertex data.
    auto& toggle_bucket = the_vertex->getToggleBucket();
    auto* toggle_data =
        toggle_bucket.frontData(PwrDataSource::kDataPropagation);
    double toggle_relative_clock =
        toggle_data ? dynamic_cast<PwrToggleData*>(toggle_data)
                          ->getToggleRateRelativeToClock()
                    : 0.0;
    node["toggle_propagation_data_relative_clock"] =
        toggle_data ? Str::printf("%.3f/clock_cycle", toggle_relative_clock)
                    : "Nil";
    double relative_clock_period_ns =
        toggle_data ? dynamic_cast<PwrToggleData*>(toggle_data)
                          ->getRelativeClockPeriodNs()
                    : 0.0;
    std::string relative_clock_name =
        toggle_data
            ? dynamic_cast<PwrToggleData*>(toggle_data)->getRelativeClockName()
            : "Nil";

    node["toggle_relative_clock_name"] =
        toggle_data ? relative_clock_name : "Nil";
    node["toggle_relative_clock_freq"] =
        toggle_data ? Str::printf("%.3fMHz", 1000 / relative_clock_period_ns)
                    : "Nil";

    node["toggle_relative_second"] =
        toggle_data ? Str::printf("%.3e/s", (1e+9 / relative_clock_period_ns) *
                                                toggle_relative_clock)
                    : "Nil";

    auto& sp_bucket = the_vertex->getSPBucket();
    auto* sp_data = sp_bucket.frontData(PwrDataSource::kDataPropagation);
    node["sp_propagation_data"] =
        sp_data ? Str::printf("%f", dynamic_cast<PwrSPData*>(sp_data)->get_sp())
                : "Nil";
  };

  print_toggle(node1, the_vertex);

  YAML::Node node2;
  vertex_node["src_arcs"] = node2;
  for (auto* src_arc : the_vertex->get_src_arcs()) {
    std::string snk_vertex_name = src_arc->get_snk()->getName();
    YAML::Node node3;
    node2[snk_vertex_name] = node3;
    print_toggle(node3, src_arc->get_snk());
  }

  YAML::Node node3;
  vertex_node["snk_arcs"] = node3;
  for (auto* snk_arc : the_vertex->get_snk_arcs()) {
    std::string src_vertex_name = snk_arc->get_src()->getName();
    YAML::Node node4;
    node3[src_vertex_name] = node4;
    print_toggle(node4, snk_arc->get_src());
  }

  return 1;
}

/**
 * @brief dump the power arc information.
 *
 * @param the_arc
 * @return unsigned
 */
unsigned PwrDumpGraphYaml::operator()(PwrArc* the_arc) { return 1; }

/**
 * @brief dump the power graph information.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned PwrDumpGraphYaml::operator()(PwrGraph* the_graph) {
  LOG_INFO << "dump graph yaml start";

  set_the_pwr_graph(the_graph);

  LOG_INFO << "dump graph node total node " << the_graph->numVertex();
  LOG_INFO << "dump graph arc total arc " << the_graph->numArc();

  PwrVertex* the_vertex;
  FOREACH_PWR_VERTEX(the_graph, the_vertex) {
    if (the_vertex->is_const()) {
      the_vertex->exec(*this);
    }
  }

  PwrArc* the_arc;
  FOREACH_PWR_ARC(the_graph, the_arc) { the_arc->exec(*this); }

  printText("graph.yaml");

  LOG_INFO << "dump graph yaml end";
  return 1;
}

/**
 * @brief print the yaml to text file.
 *
 * @param file_name
 */
void PwrDumpGraphYaml::printText(const char* file_name) {
  std::ofstream file(file_name, std::ios::trunc);
  file << _node << std::endl;
  file.close();
}

/**
 * @brief dump graphviz for the power arc.
 *
 * @param the_arc
 * @return unsigned
 */
unsigned PwrDumpGraphViz::operator()(PwrArc* the_arc) {
  std::string src_name = the_arc->get_src()->getName();
  std::string snk_name = the_arc->get_snk()->getName();

  auto* src_seq_vertex = the_arc->get_src()->get_own_seq_vertex();
  auto* snk_seq_vertex = the_arc->get_snk()->get_own_seq_vertex();

  auto new_src_name = Str::replace(src_name, "[/:\\[\\]]", "_");
  auto new_snk_name = Str::replace(snk_name, "[/:\\[\\]]", "_");

  if (src_seq_vertex) {
    _ss << new_src_name
        << Str::printf(" [label=\"%s level %d\"];\n", new_src_name.c_str(),
                       src_seq_vertex->get_level());
  }

  if (snk_seq_vertex) {
    _ss << new_snk_name
        << Str::printf(" [label=\"%s level %d\"];\n", new_snk_name.c_str(),
                       snk_seq_vertex->get_level());
  }

  _ss << new_src_name << " -> " << new_snk_name << " \n";

  return 1;
}

/**
 * @brief print the graphviz  file.
 *
 * @param file_name
 */
void PwrDumpGraphViz::printText(const char* file_name) {
  LOG_INFO << "dump graph dotviz start";

  std::ofstream dot_file;
  dot_file.open(file_name, std::ios::trunc);

  // file path begin.
  dot_file << "digraph g {"
           << "\n";

  // print arc information.
  dot_file << _ss.str();

  // file path end.
  dot_file << "}"
           << "\n";

  dot_file.close();

  LOG_INFO << "dump graph dotviz end";
}

}  // namespace ipower