// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
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

#include "api/Power.hh"
#include "core/PwrSeqGraph.hh"
#include "sta/Sta.hh"
#include "sta/StaDump.hh"
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
    if (!the_vertex->is_const()) {
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

/**
 * @brief dump the power node feature.
 *
 * @param the_graph
 * @return PwrDumpGraphJson::json
 */
PwrDumpGraphJson::json PwrDumpGraphJson::dumpNodeFeature(PwrGraph* the_graph) {
  json all_vertex_node_feature_array = json::array();

  auto* the_sta_graph = the_graph->get_sta_graph();
  auto* nl = the_sta_graph->get_nl();
  auto [die_width, die_height] = nl->get_die_size().value();

  auto& pwr_vertexes = the_graph->get_vertexes();
  for (auto& pwr_vertex : pwr_vertexes) {
    auto* the_sta_vertex = pwr_vertex->get_sta_vertex();
    json one_node_feature_array = json::array();

    auto* the_obj = the_sta_vertex->get_design_obj();
    the_obj->isPort() ? one_node_feature_array.push_back(1.0)  // is_port
                      : one_node_feature_array.push_back(0.0);
    the_obj->isInput() ? one_node_feature_array.push_back(0.0)  // is_input
                       : one_node_feature_array.push_back(1.0);
    // the distance to 4 die boundary, left, right, top, bottom TBD.
    if (the_obj->get_coordinate()) {
      auto [pin_x, pin_y] = the_obj->get_coordinate().value();
      double left_bottom_distance = pin_x + pin_y;
      double right_bottom_distance = die_width - pin_x + pin_y;
      double left_top_distance = pin_x + die_height - pin_y;
      double right_top_distance = die_width - pin_x + die_height - pin_y;

      // the order is lb(left bottom), rt, rb, lt
      one_node_feature_array.push_back(left_bottom_distance);
      one_node_feature_array.push_back(right_top_distance);
      one_node_feature_array.push_back(right_bottom_distance);
      one_node_feature_array.push_back(left_top_distance);

    } else {
      // assume the non-pin node is in the left bottom of the die.
      one_node_feature_array.push_back(0.0);
      one_node_feature_array.push_back(die_width + die_height);
      one_node_feature_array.push_back(die_width);
      one_node_feature_array.push_back(die_height);
    }

    // TODO(to taosimin), min or max first? assume min first
    double max_rise_cap =
        the_sta_vertex->getLoad(AnalysisMode::kMax, TransType::kRise);
    double max_fall_cap =
        the_sta_vertex->getLoad(AnalysisMode::kMax, TransType::kFall);
    double min_rise_cap =
        the_sta_vertex->getLoad(AnalysisMode::kMin, TransType::kRise);
    double min_fall_cap =
        the_sta_vertex->getLoad(AnalysisMode::kMin, TransType::kFall);

    one_node_feature_array.push_back(min_rise_cap);
    one_node_feature_array.push_back(min_fall_cap);

    one_node_feature_array.push_back(max_rise_cap);
    one_node_feature_array.push_back(max_fall_cap);

    double toggle_data = pwr_vertex->getToggleData(std::nullopt);
    double sp_data = pwr_vertex->getSPData(std::nullopt);

    one_node_feature_array.push_back(toggle_data);
    one_node_feature_array.push_back(sp_data);

    all_vertex_node_feature_array.push_back(one_node_feature_array);
  }

  return all_vertex_node_feature_array;
}

/**
 * @brief for net driver node, dump the net power.
 *
 * @param the_graph
 * @return PwrDumpGraphJson::json
 */
PwrDumpGraphJson::json PwrDumpGraphJson::dumpNodeNetPower(PwrGraph* the_graph) {
  json all_vertex_node_net_power_array = json::array();

  auto* ista = ista::Sta::getOrCreateSta();
  Power* ipower = Power::getOrCreatePower(&(ista->get_graph()));

  // build switch power map.
  auto& switch_powers = ipower->get_switch_powers();
  std::map<PwrVertex*, double> vertex_to_switch_power;
  for (auto& switch_power : switch_powers) {
    auto* the_net = switch_power->get_design_obj();
    auto* the_driver = dynamic_cast<ista::Net*>(the_net)->getDriver();
    auto* the_sta_vertex = ista->findVertex(the_driver);
    auto* the_pwr_vertex = the_graph->staToPwrVertex(the_sta_vertex);
    vertex_to_switch_power[the_pwr_vertex] = switch_power->get_switch_power();
  }

  // get switch power for the vertex.
  auto& pwr_vertexes = the_graph->get_vertexes();
  for (auto& pwr_vertex : pwr_vertexes) {
    double switch_power = 0.0;
    if (vertex_to_switch_power.contains(pwr_vertex.get())) {
      switch_power = vertex_to_switch_power[pwr_vertex.get()];
    }

    all_vertex_node_net_power_array.push_back(switch_power);
  }

  return all_vertex_node_net_power_array;
}

/**
 * @brief for input pin node, dump the pin internal power.
 *
 * @param the_graph
 * @return PwrDumpGraphJson::json
 */
PwrDumpGraphJson::json PwrDumpGraphJson::dumpNodeInternalPower(
    PwrGraph* the_graph) {
  json all_vertex_node_internal_power_array = json::array();
  auto& pwr_vertexes = the_graph->get_vertexes();
  for (auto& pwr_vertex : pwr_vertexes) {
    double internal_power = pwr_vertex->getInternalPower();
    all_vertex_node_internal_power_array.push_back(internal_power);
  }

  return all_vertex_node_internal_power_array;
}

/**
 * @brief for cell instance power arc, dump the arc power.
 *
 * @param the_graph
 * @return PwrDumpGraphJson::json
 */
PwrDumpGraphJson::json PwrDumpGraphJson::dumpInstInternalPower(
    PwrGraph* the_graph) {
  json all_inst_arc_delay_array = json::array();
  auto& pwr_arcs = the_graph->get_arcs();
  for (auto& the_pwr_arc : pwr_arcs) {
    if (the_pwr_arc->isInstArc()) {
      double internal_power =
          dynamic_cast<PwrInstArc*>(the_pwr_arc.get())->getInternalPower();
      all_inst_arc_delay_array.push_back(internal_power);
    }
  }

  return all_inst_arc_delay_array;
}

/**
 * @brief dump inst power arc feature.
 *
 * @param the_graph
 * @return PwrDumpGraphJson::json
 */
PwrDumpGraphJson::json PwrDumpGraphJson::dumpInstPowerArcFeature(
    PwrGraph* the_graph) {
  auto& the_arcs = the_graph->get_arcs();
  json all_inst_arc_lib_data_array = json::array();

  for (auto& the_arc : the_arcs) {
    if (the_arc->isInstArc()) {
      json one_inst_arc_table_array = json::array();
      auto* the_pwr_arc_set =
          dynamic_cast<PwrInstArc*>(the_arc.get())->get_power_arc_set();
      // for skywater130, assume only one power arc.
      auto* the_pwr_arc = the_pwr_arc_set->front();
      auto* power_model = dynamic_cast<LibPowerTableModel*>(
          the_pwr_arc->get_power_table_model());
      auto& power_tables = power_model->get_tables();

      for (auto& power_table : power_tables) {
        // copy axies
        if (power_table) {
          double is_valid = 1.0;
          one_inst_arc_table_array.push_back(is_valid);
          auto& table_axes = power_table->get_axes();
          for (auto& table_axis : table_axes) {
            auto& axis_values = table_axis->get_axis_values();
            for (auto& axis_value : axis_values) {
              double data_value = axis_value->getFloatValue();
              one_inst_arc_table_array.push_back(data_value);
            }
          }
        } else {
          double is_valid = 0.0;
          one_inst_arc_table_array.push_back(is_valid);
          // hard code 2 axis, 7*2 data
          for (int i = 0; i < 7 * 2; i++) {
            double data_value = 0.0;
            one_inst_arc_table_array.push_back(data_value);
          }
        }
      }

      // copy table values
      for (auto& power_table : power_tables) {
        if (power_table) {
          auto& table_values = power_table->get_table_values();
          for (auto& table_value : table_values) {
            one_inst_arc_table_array.push_back(table_value->getFloatValue());
          }
        } else {
          for (int i = 0; i < 7; ++i) {
            double data_value = 0.0;
            one_inst_arc_table_array.push_back(data_value);
          }
        }
      }

      all_inst_arc_lib_data_array.push_back(one_inst_arc_table_array);
    }
  }

  return all_inst_arc_lib_data_array;
}

/**
 * @brief dump the power graph json for power predict.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned PwrDumpGraphJson::operator()(PwrGraph* the_graph) {
  LOG_INFO << "dump graph json start";
  auto* the_sta_graph = the_graph->get_sta_graph();
  ista::StaDumpGraphJson dump_sta_graph_json(_json_file);

  unsigned num_nodes = the_graph->numVertex();
  _json_file["num_nodes"] = num_nodes;

  _json_file["edges"] = dump_sta_graph_json.dumpEdges(the_sta_graph);

  // dump node features
  auto n_node_features = dumpNodeFeature(the_graph);
  auto n_internal_power = dumpNodeInternalPower(the_graph);
  auto n_net_power = dumpNodeNetPower(the_graph);

  _json_file["node_features"]["nf"] = n_node_features;
  _json_file["node_features"]["n_internal_powers"] = n_internal_power;
  _json_file["node_features"]["n_net_powers"] = n_net_power;

  // dump arc features
  auto e_inst_arc_internal_power = dumpInstInternalPower(the_graph);
  auto e_net_in_arc_features =
      dump_sta_graph_json.dumpNetInArcFeature(the_sta_graph);
  auto e_net_out_arc_features =
      dump_sta_graph_json.dumpNetOutArcFeature(the_sta_graph);
  auto e_inst_pwr_arc_features = dumpInstPowerArcFeature(the_graph);

  _json_file["edge_features"]["cell_out"]["e_inst_arc_internal_power"] =
      e_inst_arc_internal_power;
  _json_file["edge_features"]["cell_out"]["ef"] = e_inst_pwr_arc_features;
  _json_file["edge_features"]["net_in"]["ef"] = e_net_in_arc_features;
  _json_file["edge_features"]["net_out"]["ef"] = e_net_out_arc_features;

  LOG_INFO << "dump graph json end";
  return 1;
}

}  // namespace ipower