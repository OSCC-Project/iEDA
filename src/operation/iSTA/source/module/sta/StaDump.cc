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

}  // namespace ista