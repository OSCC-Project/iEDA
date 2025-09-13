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
 * @file StaDump.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief Dump the Sta data
 * @version 0.1
 * @date 2021-04-22
 */

#include "StaDump.hh"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <ranges>
#include <regex>
#include <string>

#include "ThreadPool/ThreadPool.h"

namespace ista {

/**
 * @brief Dump the vertex data.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaDumpYaml::operator()(StaVertex* the_vertex) {
  YAML::Node& node = _node;

  StaVertex* vertex = the_vertex;

  static int node_id = 0;
  std::string node_name = Str::printf("node_%d", node_id++);
  YAML::Node vertex_node;
  node[node_name] = vertex_node;

  vertex_node["name"] = vertex->getNameWithCellName();
  YAML::Node node1;
  vertex_node["attribute"] = node1;
  node1["is_clock"] = vertex->is_clock();
  node1["is_port"] = vertex->is_port();
  node1["is_start"] = vertex->is_start();
  node1["is_end"] = vertex->is_end();
  node1["is_const"] = vertex->is_const();
  node1["color"] = vertex->isWhite() ? 0 : vertex->isGray() ? 1 : 2;
  node1["level"] = vertex->get_level();
  node1["is_slew_propagated"] = vertex->is_slew_prop();
  node1["is_delay_propagated"] = vertex->is_delay_prop();
  node1["is_fwd"] = vertex->is_fwd();
  node1["is_bwd"] = vertex->is_bwd();

  YAML::Node node2;
  vertex_node["src_arcs"] = node2;
  FOREACH_SRC_ARC(vertex, src_arc) {
    node2.push_back(src_arc->get_snk()->getName());
  }
  YAML::Node node3;
  vertex_node["snk_arcs"] = node3;
  FOREACH_SNK_ARC(vertex, snk_arc) {
    node3.push_back(snk_arc->get_src()->getName());
  }

  YAML::Node node4;
  vertex_node["slew_data"] = node4;
  StaData* data;
  FOREACH_SLEW_DATA(vertex, data) {
    auto* slew_data = dynamic_cast<StaSlewData*>(data);
    double slew_value = FS_TO_NS(slew_data->get_slew());
    auto delay_type = slew_data->get_delay_type();
    auto trans_type = slew_data->get_trans_type();

    auto* src_slew_data = slew_data->get_bwd();
    auto* bwd_vertex =
        src_slew_data ? src_slew_data->get_own_vertex() : nullptr;

    const char* slew_str = Str::printf(
        "%p %.3f %s %s src_vertex %s src_slew %p", slew_data, slew_value,
        delay_type == AnalysisMode::kMax ? "max" : "min",
        trans_type == TransType::kRise ? "rise" : "fall",
        bwd_vertex ? bwd_vertex->getName().c_str() : "Nil",
        src_slew_data ? src_slew_data : 0);

    node4.push_back(slew_str);
  }

  YAML::Node node5;
  vertex_node["clock_data"] = node5;
  FOREACH_CLOCK_DATA(vertex, data) {
    auto* clock_data = dynamic_cast<StaClockData*>(data);
    double arrive_time_value = FS_TO_NS(clock_data->get_arrive_time());
    auto delay_type = clock_data->get_delay_type();
    auto trans_type = clock_data->get_trans_type();

    auto* src_clock_data = clock_data->get_bwd();
    auto* bwd_vertex =
        src_clock_data ? src_clock_data->get_own_vertex() : nullptr;

    const char* data_str = Str::printf(
        "%p %.3f %s %s src_vertex %s src_data %p own clock %s", clock_data,
        arrive_time_value, delay_type == AnalysisMode::kMax ? "max" : "min",
        trans_type == TransType::kRise ? "rise" : "fall",
        bwd_vertex ? bwd_vertex->getName().c_str() : "Nil",
        src_clock_data ? src_clock_data : 0,
        clock_data->get_prop_clock()->get_clock_name());

    node5.push_back(data_str);
  }

  YAML::Node node6;
  vertex_node["path_data"] = node6;
  FOREACH_DELAY_DATA(vertex, data) {
    auto* path_delay = dynamic_cast<StaPathDelayData*>(data);
    double arrive_time_value = FS_TO_NS(path_delay->get_arrive_time());
    auto req_time_value = FS_TO_NS(path_delay->get_req_time().value_or(0));
    auto delay_type = path_delay->get_delay_type();
    auto trans_type = path_delay->get_trans_type();

    auto* src_data = path_delay->get_bwd();
    auto* src_vertex = src_data ? src_data->get_own_vertex() : nullptr;

    const char* data_str =
        Str::printf("%p AT: %.3f RT: %.3f %s %s src_vertex %s src_data %p",
                    path_delay, arrive_time_value, req_time_value,
                    delay_type == AnalysisMode::kMax ? "max" : "min",
                    trans_type == TransType::kRise ? "rise" : "fall",
                    src_vertex ? src_vertex->getName().c_str() : "Nil",
                    src_data ? src_data : 0);

    node6.push_back(data_str);
  }

  return 1;
}

/**
 * @brief Dump the arc data.
 *
 * @param the_arc
 * @return unsigned
 */
unsigned StaDumpYaml::operator()(StaArc* the_arc) {
  StaArc* arc = the_arc;
  YAML::Node& node = _node;

  static int node_id = 0;
  std::string node_name = Str::printf("arc_%d", node_id++);
  YAML::Node arc_node;
  node[node_name] = arc_node;

  arc_node["src"] = arc->get_src()->getName();
  arc_node["snk"] = arc->get_snk()->getName();
  arc_node["arc_type"] = arc->isDelayArc()
                             ? (arc->isInstArc() ? "cell Delay" : "net Delay")
                         : arc->isSetupArc() ? "Setup"
                         : arc->isHoldArc()  ? "Hold"
                                             : "Other";

  YAML::Node delay_node;
  arc_node["arc_delay_data"] = delay_node;

  StaData* data;
  FOREACH_ARC_DELAY_DATA(arc, data) {
    auto* arc_delay_data = dynamic_cast<StaArcDelayData*>(data);
    auto delay_type = arc_delay_data->get_delay_type();
    auto trans_type = arc_delay_data->get_trans_type();
    auto arc_delay_value = arc_delay_data->get_arc_delay();

    const char* arc_delay_str =
        Str::printf("%d %s %s", arc_delay_value,
                    delay_type == AnalysisMode::kMax ? "max" : "min",
                    trans_type == TransType::kRise ? "rise" : "fall");

    delay_node.push_back(arc_delay_str);
  }

  return 1;
}

/**
 * @brief Print in yaml for text
 *
 * @param the_graph
 * @return unsigned 1 if success, 0 else fail.
 */
unsigned StaDumpYaml::operator()(StaGraph* the_graph) {
  LOG_INFO << "dump graph yaml start";

  LOG_INFO << "dump graph node total node " << the_graph->numVertex();
  LOG_INFO << "dump graph arc total arc " << the_graph->numArc();

  StaVertex* the_vertex;
  FOREACH_VERTEX(the_graph, the_vertex) {
    the_vertex->exec(*this);
    LOG_INFO_EVERY_N(10000) << "dump 10000 vertexes ...";
  }

  StaArc* the_arc;
  FOREACH_ARC(the_graph, the_arc) {
    the_arc->exec(*this);
    LOG_INFO_EVERY_N(10000) << "dump 10000 arcs ...";
  }

  printText(_yaml_file_path.c_str());

  LOG_INFO << "dump graph yaml end";
  return 1;
}

/**
 * @brief Print the yaml to text.
 *
 * @param file_name
 */
void StaDumpYaml::printText(const char* file_name) {
  std::ofstream file(file_name, std::ios::trunc);
  file << _node << std::endl;
  file.close();

  LOG_INFO << "output yaml file path: " << file_name;
}

/**
 * @brief dump delay data of the vertex in yaml.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaDumpDelayYaml::operator()(StaVertex* the_vertex) {
  YAML::Node& node = _node;
  AnalysisMode analysis_mode = _analysis_mode;
  TransType trans_type = _trans_type;

  unsigned& node_id = _node_id;
  std::string node_name = Str::printf("node_%d", node_id++);
  YAML::Node vertex_node;
  node[node_name] = vertex_node;

  vertex_node["Point"] = the_vertex->getNameWithCellName();
  auto vertex_load = the_vertex->getLoad(analysis_mode, trans_type);
  vertex_node["Capacitance"] = vertex_load;
  auto vertex_slew = the_vertex->getSlewNs(analysis_mode, trans_type);
  vertex_node["slew"] = vertex_slew ? *vertex_slew : 0.0;
  vertex_node["trans_type"] =
      (trans_type == TransType::kRise) ? "rise" : "fall";

  return 1;
}

/**
 * @brief dump delay data of the arc in yaml.
 *
 * @param the_arc
 * @return unsigned
 */
unsigned StaDumpDelayYaml::operator()(StaArc* the_arc) {
  YAML::Node& node = _node;
  AnalysisMode analysis_mode = _analysis_mode;
  TransType trans_type = _trans_type;

  unsigned& arc_id = _arc_id;
  const char* arc_type_str = the_arc->isNetArc() ? "net" : "inst";
  std::string node_name = Str::printf("%s_arc_%d", arc_type_str, arc_id++);
  YAML::Node arc_node;
  node[node_name] = arc_node;

  arc_node["Incr"] =
      FS_TO_NS(the_arc->get_arc_delay(analysis_mode, trans_type));

  if (the_arc->isNetArc()) {
    auto* the_net_arc = dynamic_cast<StaNetArc*>(the_arc);
    auto* the_net = the_net_arc->get_net();

    auto* rc_net = getSta()->getRcNet(the_net);
    auto* snk_node = the_arc->get_snk();
    auto* snk_obj = snk_node->get_design_obj();
    arc_node["net_name"] = the_net->get_name();
    arc_node["fanout"] = the_net->getLoads().size();
    if (rc_net) {
      arc_node["Elmore"] =
          rc_net->delayNs(*snk_obj, RcNet::DelayMethod::kElmore).value_or(0.0);
      arc_node["D2M"] =
          rc_net->delayNs(*snk_obj, RcNet::DelayMethod::kD2M).value_or(0.0);
      arc_node["ECM"] =
          rc_net->delayNs(*snk_obj, RcNet::DelayMethod::kECM).value_or(0.0);
      arc_node["D2MC"] =
          rc_net->delayNs(*snk_obj, RcNet::DelayMethod::kD2MC).value_or(0.0);
    }
  }

  return 1;
}

/**
 * @brief dump timing vertex data for AI EDA dataset.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaDumpWireYaml::operator()(StaVertex* the_vertex) {
  AnalysisMode analysis_mode = _analysis_mode;
  TransType trans_type = _trans_type;

  unsigned& node_id = _node_id;
  std::string node_name = Str::printf("node_%d", node_id++);
  _file << node_name << ":\n";

  _file << "  Point: " << the_vertex->getNameWithCellName() << "\n";
  auto vertex_load = the_vertex->getLoad(analysis_mode, trans_type);
  _file << "  Capacitance: " << vertex_load << "\n";
  auto vertex_slew = the_vertex->getSlewNs(analysis_mode, trans_type);
  _file << "  slew: " << (vertex_slew ? *vertex_slew : 0.0) << "\n";
  _file << "  trans_type: "
        << ((trans_type == TransType::kRise) ? "rise" : "fall") << "\n";

  return 1;
}

/**
 * @brief dump timing arc data, for net arc, we need extract the wire topo, from
 * driver pin to load pin.
 * @param the_arc
 * @return unsigned
 */
unsigned StaDumpWireYaml::operator()(StaArc* the_arc) {
  AnalysisMode analysis_mode = _analysis_mode;
  TransType trans_type = _trans_type;

  auto vertex_slew = the_arc->get_src()->getSlewNs(analysis_mode, trans_type);

  unsigned& arc_id = _arc_id;
  const char* arc_type_str = the_arc->isNetArc() ? "net" : "inst";
  std::string node_name = Str::printf("%s_arc_%d", arc_type_str, arc_id++);
  _file << node_name << ":\n";

  _file << "  Incr: "
        << FS_TO_NS(the_arc->get_arc_delay(analysis_mode, trans_type)) << "\n";

  if (the_arc->isNetArc()) {
    // for net arc, we need extract the wire topo.
    auto* the_net_arc = dynamic_cast<StaNetArc*>(the_arc);
    auto* the_net = the_net_arc->get_net();

    auto* rc_net = getSta()->getRcNet(the_net);
    if ((rc_net == nullptr) || (rc_net->rct() == nullptr)) {
      return 0;
    }
    auto* snk_node = the_arc->get_snk();
    auto snk_node_name = snk_node->get_design_obj()->getFullName();

    auto wire_topo = rc_net->getWireTopo(snk_node_name.c_str());
    auto& all_nodes_slew =
        rc_net->getAllNodeSlew(*vertex_slew, analysis_mode, trans_type);
    for (int edge_index = 0;
         auto* wire_edge : wire_topo | std::ranges::views::reverse) {
      std::string edge_index_name = Str::printf("edge_%d", edge_index++);

      _file << "  " << edge_index_name << ":\n";

      auto& from_node = wire_edge->get_from();
      auto& to_node = wire_edge->get_to();

      _file << "    wire_from_node: " << from_node.get_name() << "\n";
      _file << "    wire_to_node: " << to_node.get_name() << "\n";
      _file << "    wire_R: " << wire_edge->get_res() << "\n";
      _file << "    wire_C: " << (from_node.nodeLoad() - to_node.nodeLoad())
            << "\n";
      _file << "    from_slew: " << all_nodes_slew[from_node.get_name()]
            << "\n";
      _file << "    to_slew: " << all_nodes_slew[to_node.get_name()] << "\n";
      _file << "    wire_delay: "
            << PS_TO_NS(to_node.delay() - from_node.delay()) << "\n";
    }
  }

  return 1;
}

/**
 * @brief dump timing vertex data in json.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaDumpWireJson::operator()(StaVertex* the_vertex) {
  AnalysisMode analysis_mode = _analysis_mode;
  TransType trans_type = _trans_type;

  json vertex_data;

  unsigned& node_id = _node_id;
  std::string node_name = Str::printf("node_%d", node_id++);
  vertex_data[node_name]["Point"] = the_vertex->getNameWithCellName();
  auto vertex_load = the_vertex->getLoad(analysis_mode, trans_type);
  vertex_data[node_name]["Capacitance"] = vertex_load;
  auto vertex_slew = the_vertex->getSlewNs(analysis_mode, trans_type);
  vertex_data[node_name]["slew"] = vertex_slew ? *vertex_slew : 0.0;
  vertex_data[node_name]["trans_type"] =
      (trans_type == TransType::kRise) ? "rise" : "fall";

  _parent_json.push_back(vertex_data);

  return 1;
}

/**
 * @brief dump timing arc data in json.
 *
 * @param the_arc
 * @return unsigned
 */
unsigned StaDumpWireJson::operator()(StaArc* the_arc) {
  AnalysisMode analysis_mode = _analysis_mode;
  TransType trans_type = _trans_type;

  auto vertex_slew = the_arc->get_src()->getSlewNs(analysis_mode, trans_type);

  unsigned& arc_id = _arc_id;
  const char* arc_type_str = the_arc->isNetArc() ? "net" : "inst";
  std::string arc_name = Str::printf("%s_arc_%d", arc_type_str, arc_id++);

  json arc_data;
  arc_data[arc_name]["Incr"] =
      FS_TO_NS(the_arc->get_arc_delay(analysis_mode, trans_type));
  if (the_arc->isNetArc()) {
    // for net arc, we need extract the wire topo.
    auto* the_net_arc = dynamic_cast<StaNetArc*>(the_arc);
    auto* the_net = the_net_arc->get_net();

    auto* rc_net = getSta()->getRcNet(the_net);
    if ((rc_net == nullptr) || (rc_net->rct() == nullptr)) {
      return 0;
    }
    auto* snk_node = the_arc->get_snk();
    auto snk_node_name = snk_node->get_design_obj()->getFullName();

    auto wire_topo = rc_net->getWireTopo(snk_node_name.c_str());
    auto& all_nodes_slew =
        rc_net->getAllNodeSlew(*vertex_slew, analysis_mode, trans_type);
    for (int edge_index = 0;
         auto* wire_edge : wire_topo | std::ranges::views::reverse) {
      std::string edge_index_name = Str::printf("edge_%d", edge_index++);

      auto& from_node = wire_edge->get_from();
      auto& to_node = wire_edge->get_to();

      arc_data[arc_name][edge_index_name]["wire_from_node"] =
          from_node.get_name();
      arc_data[arc_name][edge_index_name]["wire_to_node"] = to_node.get_name();
      arc_data[arc_name][edge_index_name]["wire_R"] = wire_edge->get_res();
      arc_data[arc_name][edge_index_name]["wire_C"] =
          (from_node.nodeLoad() - to_node.nodeLoad());
      arc_data[arc_name][edge_index_name]["from_slew"] =
          all_nodes_slew[from_node.get_name()];
      arc_data[arc_name][edge_index_name]["to_slew"] =
          all_nodes_slew[to_node.get_name()];
      arc_data[arc_name][edge_index_name]["wire_delay"] =
          PS_TO_NS(to_node.delay() - from_node.delay());
    }
  }

  _parent_json.push_back(arc_data);

  return 1;
}

/**
 * @brief Print Graph in GraphViz dot format.
 *
 * @param the_graph
 * @return unsigned 1 if success, 0 else fail.
 */
unsigned StaDumpGraphViz::operator()(StaGraph* the_graph) {
  LOG_INFO << "dump graph dotviz start";

  std::ofstream dot_file;
  dot_file.open("digraph.dot");

  dot_file << "digraph g {"
           << "\n";

  StaArc* arc;
  FOREACH_ARC(the_graph, arc) {
    auto src_name = arc->get_src()->getName();
    auto snk_name = arc->get_snk()->getName();

    auto new_src_name = Str::replace(src_name, "[/:\\[\\]]", "_");
    auto new_snk_name = Str::replace(snk_name, "[/:\\[\\]]", "_");

    auto max_rise_delay =
        arc->get_arc_delay(AnalysisMode::kMax, TransType::kRise);
    auto max_fall_delay =
        arc->get_arc_delay(AnalysisMode::kMax, TransType::kFall);
    auto min_rise_delay =
        arc->get_arc_delay(AnalysisMode::kMin, TransType::kRise);
    auto min_fall_delay =
        arc->get_arc_delay(AnalysisMode::kMin, TransType::kRise);

    std::string label =
        Str::printf("[label = \" ps ar %.3f af %.3f ir %.3f if %.3f\"]",
                    FS_TO_PS(max_rise_delay), FS_TO_PS(max_fall_delay),
                    FS_TO_PS(min_rise_delay), FS_TO_PS(min_fall_delay));

    dot_file << new_src_name << " -> " << new_snk_name << " " << label << "\n";
  }

  dot_file << "}"
           << "\n";

  dot_file.close();

  LOG_INFO << "dump graph dotviz end";

  return 1;
}

/**
 * @brief dump the timing data of the timing arc.
 *
 * @param the_arc
 * @return unsigned
 */
unsigned StaDumpTimingData::operator()(StaArc* the_arc) {
  AnalysisMode analysis_mode = _analysis_mode;
  TransType trans_type = _trans_type;

  auto vertex_slew = the_arc->get_src()->getSlewNs(analysis_mode, trans_type);
  if (the_arc->isInstArc()) {
    double inst_arc_delay =
        FS_TO_NS(the_arc->get_arc_delay(analysis_mode, trans_type));

    StaWireTimingData wire_timing_data;
    wire_timing_data._from_node_name = the_arc->get_src()->getName();
    wire_timing_data._to_node_name = the_arc->get_snk()->getName();
    wire_timing_data._wire_from_slew = vertex_slew.value_or(0.0);
    wire_timing_data._wire_to_slew =
        the_arc->get_snk()->getSlewNs(analysis_mode, trans_type).value_or(0.0);
    wire_timing_data._wire_delay = inst_arc_delay;

    _wire_timing_datas.emplace_back(wire_timing_data);
  } else {
    // for net arc, we need extract the wire topo.
    auto* the_net_arc = dynamic_cast<StaNetArc*>(the_arc);
    auto* the_net = the_net_arc->get_net();

    auto* rc_net = getSta()->getRcNet(the_net);
    if ((rc_net == nullptr) || (rc_net->rct() == nullptr)) {
      return 0;
    }

    auto* rc_tree = rc_net->rct();
    auto* snk_node = the_arc->get_snk();
    auto snk_node_name = snk_node->get_design_obj()->getFullName();

    auto wire_topo = rc_tree->getWireTopo(snk_node_name.c_str());
    auto all_nodes_slew =
        rc_tree->getAllNodeSlew(*vertex_slew, analysis_mode, trans_type);
    for (auto* wire_edge : wire_topo | std::ranges::views::reverse) {
      auto& from_node = wire_edge->get_from();
      auto& to_node = wire_edge->get_to();

      StaWireTimingData wire_timing_data;

      wire_timing_data._from_node_name = from_node.get_name();
      wire_timing_data._to_node_name = to_node.get_name();
      wire_timing_data._wire_resistance = wire_edge->get_res();
      wire_timing_data._wire_capacitance =
          from_node.nodeLoad() - to_node.nodeLoad();
      wire_timing_data._wire_from_slew = all_nodes_slew[from_node.get_name()];
      wire_timing_data._wire_to_slew = all_nodes_slew[to_node.get_name()];
      wire_timing_data._wire_delay =
          PS_TO_NS(to_node.delay() - from_node.delay());

      _wire_timing_datas.emplace_back(wire_timing_data);
    }
  }

  return 1;
}

/**
 * @brief dump arc in edge connection in json.
 *
 * @param the_graph
 * @return StaDumpGraphJson::json
 */
StaDumpGraphJson::json StaDumpGraphJson::dumpEdges(StaGraph* the_graph) {
  auto& the_vertexes = the_graph->get_vertexes();
  std::map<StaVertex*, int> vertex_id_map;
  int vertex_index = 0;
  for (auto& the_vertex : the_vertexes) {
    vertex_id_map[the_vertex.get()] = vertex_index++;
  }

  json edges;

  auto& the_arcs = the_graph->get_arcs();
  for (auto& the_arc : the_arcs) {
    if (the_arc->isDelayArc()) {
      int src_id = vertex_id_map[the_arc->get_src()];
      int snk_id = vertex_id_map[the_arc->get_snk()];

      if (the_arc->isInstArc() && the_arc->isDelayArc()) {
        edges["cell_out"]["src"].push_back(src_id);
        edges["cell_out"]["dst"].push_back(snk_id);
      } else if (the_arc->isNetArc()) {
        edges["net_out"]["src"].push_back(src_id);
        edges["net_out"]["dst"].push_back(snk_id);

        // reverse direction for net out
        edges["net_in"]["src"].push_back(snk_id);
        edges["net_in"]["dst"].push_back(src_id);
      }
    }
  }
  return edges;
}

/**
 * @brief dump all node require arrive time data in json.
 *
 * @param the_graph
 * @return StaDumpGraphJson::json
 */
StaDumpGraphJson::json StaDumpGraphJson::dumpNodeRAT(StaGraph* the_graph) {
  auto& the_vertexes = the_graph->get_vertexes();
  const double inf = 1.1e20;
  json all_vertex_rat_array = json::array();

  for (auto& the_vertex : the_vertexes) {
    json one_vertex_rat_array = json::array();

    double max_rise_rat =
        the_vertex->getReqTimeNs(AnalysisMode::kMax, TransType::kRise)
            .value_or(inf);
    double max_fall_rat =
        the_vertex->getReqTimeNs(AnalysisMode::kMax, TransType::kFall)
            .value_or(inf);
    double min_rise_rat =
        the_vertex->getReqTimeNs(AnalysisMode::kMin, TransType::kRise)
            .value_or(inf);
    double min_fall_rat =
        the_vertex->getReqTimeNs(AnalysisMode::kMin, TransType::kFall)
            .value_or(inf);
    // min first
    one_vertex_rat_array.push_back(min_rise_rat);
    one_vertex_rat_array.push_back(min_fall_rat);
    one_vertex_rat_array.push_back(max_rise_rat);
    one_vertex_rat_array.push_back(max_fall_rat);

    all_vertex_rat_array.push_back(one_vertex_rat_array);
  }

  return all_vertex_rat_array;
}

/**
 * @brief dump node net delay data in json.
 *
 * @param the_graph
 * @return StaDumpGraphJson::json
 */
StaDumpGraphJson::json StaDumpGraphJson::dumpNodeNetDelay(StaGraph* the_graph) {
  auto& the_vertexes = the_graph->get_vertexes();
  json all_vertex_node_net_delay_array = json::array();

  for (auto& the_vertex : the_vertexes) {
    json one_vertex_net_delay_array = json::array();

    auto* the_obj = the_vertex->get_design_obj();
    std::string obj_name = the_obj->getFullName();
    auto* the_net = the_obj->get_net();
    auto* rc_net = getSta()->getRcNet(the_net);
    RcTree* rc_tree = nullptr;
    if (rc_net) {
      rc_tree = rc_net->rct();
    }

    double max_rise_delay = 0.0;
    double max_fall_delay = 0.0;
    double min_rise_delay = 0.0;
    double min_fall_delay = 0.0;
    if (rc_tree) {
      max_rise_delay = rc_tree->delay(obj_name.c_str(), AnalysisMode::kMax,
                                      TransType::kRise);
      max_fall_delay = rc_tree->delay(obj_name.c_str(), AnalysisMode::kMax,
                                      TransType::kFall);
      min_rise_delay = rc_tree->delay(obj_name.c_str(), AnalysisMode::kMin,
                                      TransType::kRise);
      min_fall_delay = rc_tree->delay(obj_name.c_str(), AnalysisMode::kMin,
                                      TransType::kFall);
    }

    // min first
    one_vertex_net_delay_array.push_back(min_rise_delay);
    one_vertex_net_delay_array.push_back(min_fall_delay);
    one_vertex_net_delay_array.push_back(max_rise_delay);
    one_vertex_net_delay_array.push_back(max_fall_delay);

    all_vertex_node_net_delay_array.push_back(one_vertex_net_delay_array);
  }

  return all_vertex_node_net_delay_array;
}

/**
 * @brief dump all node arrive time data in json.
 *
 * @param the_graph
 * @return StaDumpGraphJson::json
 */
StaDumpGraphJson::json StaDumpGraphJson::dumpNodeAT(StaGraph* the_graph) {
  auto& the_vertexes = the_graph->get_vertexes();
  const double inf = 1.1e20;
  json all_vertex_at_array = json::array();

  for (auto& the_vertex : the_vertexes) {
    json one_vertex_at_array = json::array();

    double max_rise_at =
        the_vertex->getArriveTimeNs(AnalysisMode::kMax, TransType::kRise)
            .value_or(inf);
    double max_fall_at =
        the_vertex->getArriveTimeNs(AnalysisMode::kMax, TransType::kFall)
            .value_or(inf);
    double min_rise_at =
        the_vertex->getArriveTimeNs(AnalysisMode::kMin, TransType::kRise)
            .value_or(inf);
    double min_fall_at =
        the_vertex->getArriveTimeNs(AnalysisMode::kMin, TransType::kFall)
            .value_or(inf);

    // min first
    one_vertex_at_array.push_back(min_rise_at);
    one_vertex_at_array.push_back(min_fall_at);
    one_vertex_at_array.push_back(max_rise_at);
    one_vertex_at_array.push_back(max_fall_at);

    all_vertex_at_array.push_back(one_vertex_at_array);
  }

  return all_vertex_at_array;
}

/**
 * @brief dump all node slew data in json.
 *
 * @param the_graph
 * @return StaDumpGraphJson::json
 */
StaDumpGraphJson::json StaDumpGraphJson::dumpNodeSlew(StaGraph* the_graph) {
  auto& the_vertexes = the_graph->get_vertexes();
  const double inf = 1.1e20;
  json all_vertex_slew_array = json::array();

  for (auto& the_vertex : the_vertexes) {
    json one_vertex_slew_array = json::array();

    double max_rise_slew =
        the_vertex->getSlewNs(AnalysisMode::kMax, TransType::kRise)
            .value_or(inf);
    double max_fall_slew =
        the_vertex->getSlewNs(AnalysisMode::kMax, TransType::kFall)
            .value_or(inf);
    double min_rise_slew =
        the_vertex->getSlewNs(AnalysisMode::kMin, TransType::kRise)
            .value_or(inf);
    double min_fall_slew =
        the_vertex->getSlewNs(AnalysisMode::kMin, TransType::kFall)
            .value_or(inf);

    // min first
    one_vertex_slew_array.push_back(min_rise_slew);
    one_vertex_slew_array.push_back(min_fall_slew);
    one_vertex_slew_array.push_back(max_rise_slew);
    one_vertex_slew_array.push_back(max_fall_slew);

    all_vertex_slew_array.push_back(one_vertex_slew_array);
  }

  return all_vertex_slew_array;
}

/**
 * @brief dump the node feature data in json include is_pin_port,
 * is_fanin_out(input or output), distance to 4 die boundary, pin capacitance.
 * @ref Guo etc, DAC22 "A Timing Engine Inspired Graph Neural Network Model for
 * Pre-Routing Slack Prediction"
 *
 * @param the_graph
 * @return StaDumpGraphJson::json
 */
StaDumpGraphJson::json StaDumpGraphJson::dumpNodeFeature(StaGraph* the_graph) {
  auto& the_vertexes = the_graph->get_vertexes();
  json all_vertex_node_feature_array = json::array();
  auto* nl = the_graph->get_nl();
  auto [die_width, die_height] = nl->get_die_size().value();

  for (auto& the_vertex : the_vertexes) {
    json one_node_feature_array = json::array();
    auto* the_obj = the_vertex->get_design_obj();
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
        the_vertex->getLoad(AnalysisMode::kMax, TransType::kRise);
    double max_fall_cap =
        the_vertex->getLoad(AnalysisMode::kMax, TransType::kFall);
    double min_rise_cap =
        the_vertex->getLoad(AnalysisMode::kMin, TransType::kRise);
    double min_fall_cap =
        the_vertex->getLoad(AnalysisMode::kMin, TransType::kFall);

    one_node_feature_array.push_back(min_rise_cap);
    one_node_feature_array.push_back(min_fall_cap);

    one_node_feature_array.push_back(max_rise_cap);
    one_node_feature_array.push_back(max_fall_cap);

    all_vertex_node_feature_array.push_back(one_node_feature_array);
  }
  return all_vertex_node_feature_array;
}

/**
 * @brief dump the node is_end_point data in json.
 *
 * @param the_graph
 * @return StaDumpGraphJson::json
 */
StaDumpGraphJson::json StaDumpGraphJson::dumpNodeIsEndPoint(
    StaGraph* the_graph) {
  auto& the_vertexes = the_graph->get_vertexes();
  json all_vertex_node_is_ep_array = json::array();

  for (auto& the_vertex : the_vertexes) {
    all_vertex_node_is_ep_array.push_back(the_vertex->is_end() ? 1.0 : 0.0);
  }
  return all_vertex_node_is_ep_array;
}

/**
 * @brief dump all instance arc delay data in json.
 *
 * @param the_graph
 * @return StaDumpGraphJson::json
 */
StaDumpGraphJson::json StaDumpGraphJson::dumpInstArcDelay(StaGraph* the_graph) {
  auto& the_arcs = the_graph->get_arcs();
  json all_inst_arc_delay_array = json::array();

  for (auto& the_arc : the_arcs) {
    if (the_arc->isInstArc() && the_arc->isDelayArc()) {
      json one_inst_arc_delay_array = json::array();

      double max_rise_delay = FS_TO_NS(
          the_arc->get_arc_delay(AnalysisMode::kMax, TransType::kRise));
      double max_fall_delay = FS_TO_NS(
          the_arc->get_arc_delay(AnalysisMode::kMax, TransType::kFall));
      double min_rise_delay = FS_TO_NS(
          the_arc->get_arc_delay(AnalysisMode::kMin, TransType::kRise));
      double min_fall_delay = FS_TO_NS(
          the_arc->get_arc_delay(AnalysisMode::kMin, TransType::kFall));

      // min first
      one_inst_arc_delay_array.push_back(min_rise_delay);
      one_inst_arc_delay_array.push_back(min_fall_delay);
      one_inst_arc_delay_array.push_back(max_rise_delay);
      one_inst_arc_delay_array.push_back(max_fall_delay);

      all_inst_arc_delay_array.push_back(one_inst_arc_delay_array);
    }
  }

  return all_inst_arc_delay_array;
}

/**
 * @brief dump inst arc lib table data in json.
 * @ref Guo etc, DAC22 "A Timing Engine Inspired Graph Neural Network Model for
 * Pre-Routing Slack Prediction"
 * @param the_graph
 * @return StaDumpGraphJson::json
 */
StaDumpGraphJson::json StaDumpGraphJson::dumpInstArcFeature(
    StaGraph* the_graph) {
  auto& the_arcs = the_graph->get_arcs();
  json all_inst_arc_lib_data_array = json::array();

  for (auto& the_arc : the_arcs) {
    if (the_arc->isInstArc() && the_arc->isDelayArc()) {
      json one_inst_arc_table_array = json::array();

      auto* the_lib_arc =
          dynamic_cast<StaInstArc*>(the_arc.get())->get_lib_arc();
      auto* delay_model =
          dynamic_cast<LibDelayTableModel*>(the_lib_arc->get_table_model());
      auto& delay_tables = delay_model->get_tables();

      std::vector<LibTable*> store_tables;
      int duplicate = 2;  // hard code 2 table duplicate for 2 corner
      for (int i = 0; i < duplicate; ++i) {
        for (auto& delay_table : delay_tables) {
          store_tables.push_back(delay_table.get());
        }
      }

      for (auto* delay_table : store_tables) {
        // copy axies
        if (delay_table) {
          double is_valid = 1.0;
          one_inst_arc_table_array.push_back(is_valid);
          auto& table_axes = delay_table->get_axes();
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
          for (int i = 0; i < 7 * 2; ++i) {
            double data_value = 0.0;
            one_inst_arc_table_array.push_back(data_value);
          }
        }
      }

      // copy table values
      for (auto* delay_table : store_tables) {
        if (delay_table) {
          auto& table_values = delay_table->get_table_values();
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
 * @brief dump net arc feature, the Manhattan distance between the positions of
a net’s drive pin and its sink pin.(driver -> sink)
 * @ref Guo etc, DAC22 "A Timing Engine Inspired Graph Neural Network Model for
 * Pre-Routing Slack Prediction"
 * @param the_graph
 * @return StaDumpGraphJson::json
 */
StaDumpGraphJson::json StaDumpGraphJson::dumpNetInArcFeature(
    StaGraph* the_graph) {
  auto& the_arcs = the_graph->get_arcs();
  json all_net_arc_feature_array = json::array();

  for (auto& the_arc : the_arcs) {
    if (the_arc->isNetArc()) {
      json one_net_arc_feature_array = json::array();

      auto* the_net_arc = dynamic_cast<StaNetArc*>(the_arc.get());
      auto* src = the_net_arc->get_src();
      auto* snk = the_net_arc->get_snk();

      auto* src_obj = src->get_design_obj();
      auto* snk_obj = snk->get_design_obj();

      auto src_coord = src_obj->get_coordinate();
      auto snk_coord = snk_obj->get_coordinate();

      double distance_x = src_coord->first - snk_coord->first;
      double distance_y = src_coord->second - snk_coord->second;

      one_net_arc_feature_array.push_back(distance_x);
      one_net_arc_feature_array.push_back(distance_y);

      all_net_arc_feature_array.push_back(one_net_arc_feature_array);
    }
  }

  return all_net_arc_feature_array;
}

/**
 * @brief The net out is net arc reverse, from sink->driver.
 *
 * @param the_graph
 * @return StaDumpGraphJson::json
 */
StaDumpGraphJson::json StaDumpGraphJson::dumpNetOutArcFeature(
    StaGraph* the_graph) {
  auto& the_arcs = the_graph->get_arcs();
  json all_net_arc_feature_array = json::array();

  for (auto& the_arc : the_arcs) {
    if (the_arc->isNetArc()) {
      json one_net_arc_feature_array = json::array();

      auto* the_net_arc = dynamic_cast<StaNetArc*>(the_arc.get());
      auto* src = the_net_arc->get_src();
      auto* snk = the_net_arc->get_snk();

      auto* src_obj = src->get_design_obj();
      auto* snk_obj = snk->get_design_obj();

      auto src_coord = src_obj->get_coordinate();
      auto snk_coord = snk_obj->get_coordinate();

      double distance_x = snk_coord->first - src_coord->first;
      double distance_y = snk_coord->second - src_coord->second;

      one_net_arc_feature_array.push_back(distance_x);
      one_net_arc_feature_array.push_back(distance_y);

      all_net_arc_feature_array.push_back(one_net_arc_feature_array);
    }
  }

  return all_net_arc_feature_array;
}

/**
 * @brief dump the graph json for get graph timing data.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned StaDumpGraphJson::operator()(StaGraph* the_graph) {
  LOG_INFO << "dump graph json start";

  unsigned num_nodes = the_graph->numVertex();
  _json_file["num_nodes"] = num_nodes;

  _json_file["edges"] = dumpEdges(the_graph);

  // dump node features
  auto n_rats = dumpNodeRAT(the_graph);
  auto n_net_delays = dumpNodeNetDelay(the_graph);
  auto n_ats = dumpNodeAT(the_graph);
  auto n_slews = dumpNodeSlew(the_graph);
  auto n_node_features = dumpNodeFeature(the_graph);
  auto n_is_timing_endpt = dumpNodeIsEndPoint(the_graph);

  _json_file["node_features"]["n_rats"] = n_rats;
  _json_file["node_features"]["n_net_delays"] = n_net_delays;
  _json_file["node_features"]["n_ats"] = n_ats;
  _json_file["node_features"]["n_slews"] = n_slews;
  _json_file["node_features"]["nf"] = n_node_features;
  _json_file["node_features"]["n_is_timing_endpt"] = n_is_timing_endpt;

  // dump arc features
  auto e_inst_arc_delays = dumpInstArcDelay(the_graph);
  auto e_inst_arc_features = dumpInstArcFeature(the_graph);
  auto e_net_in_arc_features = dumpNetInArcFeature(the_graph);
  auto e_net_out_arc_features = dumpNetOutArcFeature(the_graph);

  _json_file["edge_features"]["cell_out"]["e_cell_delays"] = e_inst_arc_delays;
  _json_file["edge_features"]["cell_out"]["ef"] = e_inst_arc_features;
  _json_file["edge_features"]["net_in"]["ef"] = e_net_in_arc_features;
  _json_file["edge_features"]["net_out"]["ef"] = e_net_out_arc_features;

  LOG_INFO << "dump graph json end";

  return 1;
}

}  // namespace ista