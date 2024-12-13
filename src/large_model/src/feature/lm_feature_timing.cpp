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
#include "lm_feature_timing.h"
#include "omp.h"
#include "usage.hh"

namespace ilm {

void LmFeatureTiming::buildWireTimingPowerFeature(LmNet* lm_net,
                                                  const std::string& net_name) {
  auto* eval_tp = ieval::InitSTA::getInst();
  auto [toggle, voltage] = eval_tp->getNetToggleAndVoltage(net_name);

  for (auto& wire : lm_net->get_wires()) {
    auto* wire_feature = wire.get_feature(true);
    auto [src, snk] = wire.get_connected_nodes();

    std::string node_name;
    int pin_id = snk->get_node_data()->get_pin_id();
    if (pin_id != -1) {
      auto [inst_name, pin_type_name] = _layout->findPinName(pin_id);
      auto pin_name = !inst_name.empty() ? (inst_name + ":" + pin_type_name)
                                         : pin_type_name;
      node_name = pin_name;
    } else {
      node_name = net_name + ":" + std::to_string(snk->get_node_id());
    }

    double resistance = eval_tp->getWireResistance(net_name, node_name);
    double capacitance = eval_tp->getWireCapacitance(net_name, node_name);
    double slew = eval_tp->getWireSlew(net_name, node_name);
    double delay = eval_tp->getWireDelay(net_name, node_name);
    double power = 0.5 * toggle * voltage * capacitance;

    LOG_INFO << "node " << node_name << " resistance " << resistance << " cap "
             << capacitance << " slew " << slew << " delay " << delay
             << " power " << power;

    wire_feature->R = resistance;
    wire_feature->C = capacitance;
    wire_feature->slew = slew;
    wire_feature->delay = delay;
    wire_feature->power = power;
  }
}

void LmFeatureTiming::buildNetTimingPowerFeature() {
  auto* eval_tp = ieval::InitSTA::getInst();

  auto& lm_graph = _layout->get_graph();
  auto& net_id_map = _layout->get_net_name_map();
  for (auto [net_name, net_id] : net_id_map) {
    LOG_INFO << "build net " << net_name << " feature ";
    auto* lm_net = lm_graph.get_net(net_id);
    auto* net_feature = lm_net->get_feature(true);

    double resistance = eval_tp->getNetResistance(net_name);
    double capacitance = eval_tp->getNetCapacitance(net_name);
    double slew = eval_tp->getNetSlew(net_name);
    double delay = eval_tp->getNetDelay(net_name);
    double power = eval_tp->getNetPower(
        net_name);  // TODO(to taosimin), get net power from eval.

    buildWireTimingPowerFeature(lm_net, net_name);

    LOG_INFO << "net " << net_name << " resistance " << resistance << " cap "
             << capacitance << " slew " << slew << " delay " << delay
             << " power " << power;

    net_feature->R = resistance;
    net_feature->C = capacitance;
    net_feature->slew = slew;
    net_feature->delay = delay;
    net_feature->power = power;
  }
}

void LmFeatureTiming::build() {
  auto* eval_tp = ieval::InitSTA::getInst();  // evaluate timing and power.

  eval_tp->runLmSTA(_layout);
  //   eval_tp->evalTiming("WireGraph", true);

  buildNetTimingPowerFeature();
}

}  // namespace ilm