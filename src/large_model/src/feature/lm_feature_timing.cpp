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

#include "lm_feature_timing.h"

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

namespace ilm {

void LmFeatureTiming::buildWireTimingPowerFeature(LmNet* lm_net, const std::string& net_name)
{
  auto* eval_tp = ieval::InitSTA::getInst();

  for (auto& wire : lm_net->get_wires()) {
    auto* wire_feature = wire.get_feature(true);
    auto [src, snk] = wire.get_connected_nodes();

    std::string node_name = net_name + ":" + std::to_string(snk->get_node_id());

    wire_feature->slew = eval_tp->getWireSlew(net_name, node_name);
    wire_feature->delay = eval_tp->getWireDelay(net_name, node_name);
    wire_feature->power = 0.0;  // TODO(to taosimin), get wire power from eval.
  }
}

void LmFeatureTiming::buildNetTimingPowerFeature()
{
  auto* eval_tp = ieval::InitSTA::getInst();

  auto& lm_graph = _layout->get_graph();
  auto& net_id_map = _layout->get_net_name_map();
  for (auto [net_name, net_id] : net_id_map) {
    LOG_INFO << "net name" << net_name;
    auto* lm_net = lm_graph.get_net(net_id);
    auto* net_feature = lm_net->get_feature(true);

    net_feature->slew = eval_tp->getNetSlew(net_name);
    net_feature->delay = eval_tp->getNetDelay(net_name);
    net_feature->power = eval_tp->getNetPower(net_name);

    buildWireTimingPowerFeature(lm_net, net_name);
  }
}

void LmFeatureTiming::build()
{
  auto* eval_tp = ieval::InitSTA::getInst();  // evaluate timing and power.

  eval_tp->runLmSTA(_layout);
  //   eval_tp->evalTiming("WireGraph", true);

  buildNetTimingPowerFeature();
}

}  // namespace ilm