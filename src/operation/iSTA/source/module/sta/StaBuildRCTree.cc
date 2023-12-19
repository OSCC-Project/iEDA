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
 * @file StaBuildRCTree.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of build rc tree.
 * @version 0.1
 * @date 2021-04-14
 */
// #include <gperftools/profiler.h>

#include "StaBuildRCTree.hh"

#include <string>
#include <utility>

#include "ThreadPool/ThreadPool.h"
#include "delay/ArnoldiDelayCal.hh"
#include "delay/ElmoreDelayCalc.hh"
#include "log/Log.hh"
#include "netlist/Netlist.hh"

namespace ista {

StaBuildRCTree::StaBuildRCTree(std::string&& spef_file_name,
                               DelayCalcMethod calc_method)
    : _spef_file_name(std::move(spef_file_name)), _calc_method(calc_method) {}

/**
 * @brief Create the rc net for delay calculation.
 *
 * @param rc_net_name net name.
 * @param net
 * @return std::unique_ptr<RcNet>
 */
std::unique_ptr<RcNet> StaBuildRCTree::createRcNet(Net* net) {
  const char* net_name = net->get_name();
  std::string rc_net_name = net_name;
  std::unique_ptr<RcNet> rc_net;
  if (_calc_method == DelayCalcMethod::kArnoldi) {
    rc_net = std::make_unique<ArnoldiNet>(net);
  } else {
    rc_net = std::make_unique<RcNet>(net);
  }
  return rc_net;
}

/**
 * @brief Build the net rc tree connected to the vertex.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaBuildRCTree::operator()(StaGraph* the_graph) {
  LOG_INFO << "build rc tree start";

  LOG_INFO << "read spef " << _spef_file_name << " start";
  spef::Spef parser;
  if (!parser.read(_spef_file_name)) {
    LOG_FATAL << "Parse the spef file error.";
    return 0;
  }

  LOG_INFO << "read spef " << _spef_file_name << " end";

  LOG_INFO << "expand spef name start";
  unsigned num_threads = getNumThreads();
  parser.expand_name(num_threads);
  LOG_INFO << "expand spef name end";

  auto rc_net_common_info = std::make_unique<RCNetCommonInfo>();
  rc_net_common_info->set_spef_cap_unit(parser.capacitance_unit);
  rc_net_common_info->set_spef_resistance_unit(parser.resistance_unit);
  RcNet::set_rc_net_common_info(std::move(rc_net_common_info));

  // ProfilerStart("rc_tree.prof");
  unsigned is_ok = 1;

  // build rc net
  Netlist* design_nl = the_graph->get_nl();
  Net* net;
  FOREACH_NET(design_nl, net) {
    auto rc_net = createRcNet(net);

    // DLOG_INFO << net->get_name() << "build rc tree";
    getSta()->addRcNet(net, std::move(rc_net));
  }

  // rc net update timing information.
  auto& spef_nets = parser.getNets();
  std::atomic<unsigned> max_node = 0;
  std::string net_name;

#if 1
  {
    ThreadPool pool(num_threads);

    for (const auto& spef_net : spef_nets) {
      // record max node net name.
      if (spef_net.caps.size() > max_node) {
        max_node = spef_net.caps.size();
        net_name = spef_net.name;
      }

      // enqueue and store future
      pool.enqueue(
          [design_nl, this](const auto& spef_net) {
            auto spef_name = spef_net.name;
            auto* design_net = design_nl->findNet(spef_name.c_str());
            if (design_net) {
              auto* rc_net = getSta()->getRcNet(design_net);
              // DLOG_INFO << "Update Rc tree timing " << spef_name;
              rc_net->updateRcTiming(spef_net);
            } else {
              LOG_FATAL << "build rc tree not found design net " << spef_name;
            }
          },
          spef_net);
    }
  }
#else
  for (auto& spef_net : spef_nets) {
    if (spef_net.caps.size() > max_node) {
      max_node = spef_net.caps.size();
      net_name = spef_net.name;
    }

    std::string spef_name = spef_net.name;
    auto* design_net = design_nl->findNet(spef_name.c_str());
    if (design_net) {
      auto* rc_net = getSta()->getRcNet(design_net);
      // DLOG_INFO << "Update Rc tree timing " << spef_name;
      rc_net->updateRcTiming(spef_net);
      // printYaml(spef_net);
    } else {
      LOG_FATAL << "build rc tree not found design net " << spef_name;
    }
  }

#endif

  LOG_INFO << "net name " << net_name << " max node " << max_node;

  // ProfilerStop();

  // printYamlText("spef_1W.yaml");

  LOG_INFO << "build rc tree end";

  return is_ok;
}

/**
 * @brief print rc tree in yaml format.
 *
 * @param top_node
 * @param spef_net
 */
void StaBuildRCTree::printYaml(const spef::Net& spef_net) {
  static int node_id = 0;
  static std::map<int, std::string> layer_index_to_name{
      {1, "M1"},    {2, "M2"},    {3, "M3"},    {4, "M4"},    {5, "M5"},
      {6, "M6"},    {7, "M7"},    {8, "M8"},    {9, "M9"},    {10, "AP"},
      {11, "VIA1"}, {12, "VIA2"}, {13, "VIA3"}, {14, "VIA4"}, {15, "VIA5"},
      {16, "VIA6"}, {17, "VIA7"}, {18, "VIA8"}, {19, "RV"}};

  std::string net_id = Str::printf("net_%d", node_id++);

  YAML::Node net_node;
  _top_node[net_id] = net_node;

  net_node["NAME"] = spef_net.name;

  // print connection.
  auto& spef_connections = spef_net.connections;
  YAML::Node conn_node;
  net_node["CONN"] = conn_node;
  for (unsigned i = 0; const auto& spef_connection : spef_connections) {
    std::string conn_id = Str::printf("conn_%d", i++);
    YAML::Node one_node;
    conn_node[conn_id] = one_node;
    one_node["type"] =
        (spef_connection.type == spef::ConnectionType::INTERNAL) ? "I" : "P";
    one_node["node_name"] = spef_connection.name;
    one_node["direction"] =
        (spef_connection.direction == spef::ConnectionDirection::OUTPUT)
            ? "O"
            : (spef_connection.direction == spef::ConnectionDirection::INPUT)
                  ? "I"
                  : "IO";
    one_node["coordinate_x"] = spef_connection.coordinate.has_value()
                                   ? spef_connection.coordinate.value().first
                                   : 0;
    one_node["coordinate_y"] = spef_connection.coordinate.has_value()
                                   ? spef_connection.coordinate.value().second
                                   : 0;
    one_node["load"] =
        spef_connection.load.has_value() ? spef_connection.load.value() : 0;
    one_node["cell"] = spef_connection.driving_cell.empty()
                           ? "NA"
                           : spef_connection.driving_cell;

    one_node["ll_coordinate_x"] =
        spef_connection.ll_coordinate.has_value()
            ? spef_connection.ll_coordinate.value().first
            : 0;
    one_node["ll_coordinate_y"] =
        spef_connection.ll_coordinate.has_value()
            ? spef_connection.ll_coordinate.value().second
            : 0;

    one_node["ur_coordinate_x"] =
        spef_connection.ur_coordinate.has_value()
            ? spef_connection.ur_coordinate.value().first
            : 0;
    one_node["ur_coordinate_y"] =
        spef_connection.ur_coordinate.has_value()
            ? spef_connection.ur_coordinate.value().second
            : 0;

    one_node["layer"] = spef_connection.layer.has_value()
                            ? layer_index_to_name[spef_connection.layer.value()]
                            : "Nil";
  }

  // print cap.
  YAML::Node cap_node;
  net_node["CAP"] = cap_node;
  auto& spef_caps = spef_net.caps;
  for (unsigned i = 0; const auto& [node1, node2, cap] : spef_caps) {
    std::string cap_id = Str::printf("cap_%d", i++);
    YAML::Node one_node;
    cap_node[cap_id] = one_node;
    one_node["node1_name"] = node1;
    one_node["node2_name"] = node2.empty() ? "NA" : node2;
    one_node["capacitance"] = cap;
  }

  // print resistance
  YAML::Node resistance_node;
  net_node["RES"] = resistance_node;
  for (unsigned i = 0; const auto& [node1, node2, res] : spef_net.ress) {
    std::string res_id = Str::printf("resistance_%d", i++);
    YAML::Node one_node;
    resistance_node[res_id] = one_node;
    one_node["node1_name"] = node1;
    one_node["node2_name"] = node2.empty() ? "NA" : node2;
    one_node["resistance"] = res;
  }
}

/**
 * @brief print yaml text.
 *
 * @param file_name
 */
void StaBuildRCTree::printYamlText(const char* file_name) {
  std::ofstream file(file_name, std::ios::trunc);
  file << _top_node << std::endl;
  file.close();
}

}  // namespace ista
