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

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Log.hh"
#include "idm.h"
#include "init_sta.hh"
#include "omp.h"
#include "usage.hh"
#include "vec_feature_timing.h"

namespace ivec {

void VecFeatureTiming::build()
{
  auto* eval_tp = ieval::InitSTA::getInst();  // evaluate timing and power.

  if (_is_placement_mode) {
    // In placement mode, use EGR estimation for timing calculation
    eval_tp->runPlaceVecSTA("EGR", false, _dir);
  } else {
    // In routing mode, user can choose between VEC_STA and SPEF_STA
    switch (_sta_mode) {
      case 0:
        // Use actual wire information from layout (VEC_STA mode)
        eval_tp->runVecSTA(_layout, _dir);
        buildNetTimingPowerFeature();
        break;
      case 1:
        // Use SPEF file for timing analysis (SPEF_STA mode)
        eval_tp->runSpefVecSTA(_dir);
        break;
      default:
        // Default to VEC_STA mode
        eval_tp->runVecSTA(_layout, _dir);
        break;
    }
  }

  auto timing_wire_graph = eval_tp->getTimingWireGraph();

  std::string yaml_graph_path = _dir + "/wire_graph";

  if (!std::filesystem::exists(yaml_graph_path)) {
    std::filesystem::create_directories(yaml_graph_path);
  }

  std::string yaml_graph_file = yaml_graph_path + "/timing_wire_graph.json";
  SaveTimingGraph(timing_wire_graph, yaml_graph_file);

  /// save timing instance graph.
  auto timing_instance_graph = eval_tp->getTimingInstanceGraph();

  yaml_graph_path = _dir + "/instance_graph";

  if (!std::filesystem::exists(yaml_graph_path)) {
    std::filesystem::create_directories(yaml_graph_path);
  }


  yaml_graph_file = yaml_graph_path + "/timing_instance_graph.json";
  SaveTimingInstanceGraph(timing_instance_graph, yaml_graph_file);
}

void VecFeatureTiming::buildWireTimingPowerFeature(VecNet* vec_net, const std::string& net_name)
{
  auto* eval_tp = ieval::InitSTA::getInst();
  auto [toggle, voltage] = eval_tp->getNetToggleAndVoltage(net_name);
  auto all_node_slews = eval_tp->getAllNodesSlew(net_name);

  auto get_node_feature = [this, eval_tp, toggle, voltage, &all_node_slews](
                              const auto& net_name, auto& vec_node) -> std::tuple<double, double, double, double, double> {
    std::string node_name;
    int pin_id = vec_node->get_node_data()->get_pin_id();
    if (pin_id != -1) {
      auto [inst_name, pin_type_name] = _layout->findPinName(pin_id);
      auto pin_name = !inst_name.empty() ? (inst_name + ":" + pin_type_name) : pin_type_name;
      node_name = pin_name;
    } else {
      node_name = net_name + ":" + std::to_string(vec_node->get_node_id());
    }

    node_name = ieda::Str::replace(node_name, R"(\\)", "");

    double resistance = eval_tp->getWireResistance(net_name, node_name);
    double capacitance = eval_tp->getWireCapacitance(net_name, node_name);
    double slew = all_node_slews.contains(node_name) ? all_node_slews.at(node_name) : 0.0;
    double delay = eval_tp->getWireDelay(net_name, node_name);
    double power = 0.5 * toggle * voltage * capacitance;

    LOG_INFO_EVERY_N(100) << "node " << node_name << " resistance " << resistance << " cap " << capacitance << " slew " << slew << " delay "
                          << delay << " power " << power;

    return std::tuple(resistance, capacitance, slew, delay, power);
  };

  for (auto& wire : vec_net->get_wires()) {
    auto* wire_feature = wire.get_feature(true);
    auto [src, snk] = wire.get_connected_nodes();

    auto [src_R, src_C, src_slew, src_delay, src_power] = get_node_feature(net_name, src);
    auto [snk_R, snk_C, snk_slew, snk_delay, snk_power] = get_node_feature(net_name, snk);

    double wire_resistance = std::abs(snk_R - src_R);
    double wire_capacitance = std::abs(snk_C - src_C);
    double wire_slew = std::abs(snk_slew - src_slew);
    double wire_delay = std::abs(snk_delay - src_delay);
    double wire_power = std::abs(snk_power - src_power);

    wire_feature->R = wire_resistance;
    wire_feature->C = wire_capacitance;
    wire_feature->slew = wire_slew;
    wire_feature->delay = wire_delay;
    wire_feature->power = wire_power;
  }
}

void VecFeatureTiming::buildNetTimingPowerFeature()
{
  auto* eval_tp = ieval::InitSTA::getInst();

  auto& vec_graph = _layout->get_graph();
  auto& net_id_map = _layout->get_net_name_map();

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < (int) net_id_map.size(); ++i) {
    auto it = net_id_map.begin();
    std::advance(it, i);
    auto& net_name = it->first;
    auto& net_id = it->second;

    if (!eval_tp->getRcNet(net_name)) {
      continue;
    }

    LOG_INFO << "build net " << net_name << " feature ";
    auto* vec_net = vec_graph.get_net(net_id);
    auto* net_feature = vec_net->get_feature(true);

    double resistance = eval_tp->getNetResistance(net_name);
    double capacitance = eval_tp->getNetCapacitance(net_name);
    double slew = eval_tp->getNetSlew(net_name);
    double delay = eval_tp->getNetDelay(net_name);
    double power = eval_tp->getNetPower(net_name);  // TODO(to taosimin), get net power from eval.

    buildWireTimingPowerFeature(vec_net, net_name);

    LOG_INFO << "net " << net_name << " resistance " << resistance << " cap " << capacitance << " slew " << slew << " delay " << delay
             << " power " << power;

    net_feature->R = resistance;
    net_feature->C = capacitance;
    net_feature->slew = slew;
    net_feature->delay = delay;
    net_feature->power = power;
  }
}

}  // namespace ivec