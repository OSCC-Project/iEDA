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

#include "EstimateParasitics.h"

#include "ToConfig.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "data_manager.h"
#include "timing_engine.h"

namespace ito {

EstimateParasitics* EstimateParasitics::_instance = nullptr;

EstimateParasitics* EstimateParasitics::get_instance()
{
  static std::mutex mt;
  if (_instance == nullptr) {
    std::lock_guard<std::mutex> lock(mt);
    if (_instance == nullptr) {
      _instance = new EstimateParasitics();
    }
  }
  return _instance;
}

void EstimateParasitics::destroy_instance()
{
  if (_instance != nullptr) {
    delete _instance;
    _instance = nullptr;
  }
}

EstimateParasitics::EstimateParasitics()
{
  Flute::readLUT();
}

/**
 * @brief If parasitics have been evaluated, update the changed net stored in
 * _parasitics_invalid_nets. else update rc tree for all net
 *
 */
void EstimateParasitics::excuteParasiticsEstimate()
{
  if (_have_estimated_parasitics) {
    for (Net* net : _parasitics_invalid_nets) {
      DesignObject* driver = net->getDriver();
      if (driver) {
        if (timingEngine->get_sta_engine()->get_ista()->getRcNet(net)) {
          timingEngine->get_sta_engine()->resetRcTree(net);
        }
        excuteWireParasitic(net);
      }
    }
    _parasitics_invalid_nets.clear();
  } else {
    estimateAllNetParasitics();
  }
}

/**
 * @brief update rc tree for all net
 *
 */
void EstimateParasitics::estimateAllNetParasitics()
{
  LOG_INFO << "estimate all net parasitics start";
  Netlist* design_nl = timingEngine->get_sta_engine()->get_netlist();
  Net* net;
  FOREACH_NET(design_nl, net)
  {
    estimateNetParasitics(net);
  }
  _have_estimated_parasitics = true;
  _parasitics_invalid_nets.clear();
  LOG_INFO << "estimate all net parasitics end";
}

/**
 * @brief update rc for special net
 *
 * @param net
 */
void EstimateParasitics::estimateNetParasitics(Net* net)
{
  if (timingEngine->get_sta_engine()->get_ista()->getRcNet(net)) {
    timingEngine->get_sta_engine()->resetRcTree(net);
  }

  excuteWireParasitic(net);
}

/**
 * @brief Re-estimate net with invalid RC values
 *
 * @param driver_pin_port
 * @param net
 */
void EstimateParasitics::estimateInvalidNetParasitics(Net* net, DesignObject* driver_pin_port)
{
  if (_parasitics_invalid_nets.find(net) != _parasitics_invalid_nets.end() && net) {

    excuteWireParasitic(net);

    _parasitics_invalid_nets.erase(net);
  }
}

void EstimateParasitics::excuteWireParasitic(Net* curr_net)
{
  TreeBuild* tree = new TreeBuild();
  bool make_tree = tree->makeRoutingTree(curr_net, toConfig->get_routing_tree());
  if (!make_tree) {
    return;
  }
  // cout << tree;

  if (timingEngine->get_sta_engine()->get_ista()->getRcNet(curr_net)) {
    timingEngine->get_sta_engine()->resetRcTree(curr_net);
  }

  vector<int> segment_idx;
  vector<int> length_wire;
  tree->driverToLoadLength(tree->get_root(), segment_idx, length_wire, 0);

  std::vector<std::pair<int, int>> wire_segment_idx;
  std::vector<int> length_per_wire;

  tree->segmentIndexAndLength(tree->get_root(), wire_segment_idx, length_per_wire);

  for (int i = 0; i < (int) wire_segment_idx.size(); ++i) {
    updateParastic(curr_net, wire_segment_idx[i].first, wire_segment_idx[i].second, length_per_wire[i], tree);
  }

  timingEngine->get_sta_engine()->updateRCTreeInfo(curr_net);

  delete tree;
}

void EstimateParasitics::updateParastic(Net* curr_net, int index1, int index2, int length_per_wire, TreeBuild* tree)
{
  RctNode* node1 = timingEngine->get_sta_engine()->makeOrFindRCTreeNode(curr_net, index1);
  RctNode* node2 = timingEngine->get_sta_engine()->makeOrFindRCTreeNode(curr_net, index2);

  if (length_per_wire == 0) {
    timingEngine->refineRes(node1, node2, curr_net);
  } else {
    std::optional<double> width = std::nullopt;
    double wire_len_cap = timingEngine->get_sta_adapter()->getCapacitance(1, (double) length_per_wire / toDmInst->get_dbu(), width);
    double wire_len_res = timingEngine->get_sta_adapter()->getResistance(1, (double) length_per_wire / toDmInst->get_dbu(), width);

    if (curr_net->isClockNet()) {
      wire_len_cap /= 10.0;
      wire_len_res /= 10.0;
    }

    timingEngine->refineRes(node1, node2, curr_net, wire_len_res, true, wire_len_cap);
  }

  /// reconnect pins
  RctNodeConnectPins(index1, node1, index2, node2, curr_net, tree);
}

void EstimateParasitics::RctNodeConnectPins(int index1, RctNode* node1, int index2, RctNode* node2, Net* net, TreeBuild* tree)
{
  auto pin_con = [](Net* net, int index, RctNode* rcnode, TreeBuild* tree) {
    int num_pins = tree->get_pins().size();
    if (tree->get_pin_visit(index) == 1) {
      return;
    }
    if (index < num_pins) {
      tree->set_pin_visit(index);
      RctNode* pin_node = timingEngine->get_sta_engine()->makeOrFindRCTreeNode(tree->get_pin(index));
      if (index == tree->get_root()->get_id()) {
        timingEngine->refineRes(pin_node, rcnode, net);
      } else {
        timingEngine->refineRes(rcnode, pin_node, net);
      }
    }
  };

  pin_con(net, index1, node1, tree);
  pin_con(net, index2, node2, tree);
}

void EstimateParasitics::invalidNetRC(Net* net)
{
  // printf("EstimateParasitics | parasitics invalid {%s}\n", net->get_name());
  _parasitics_invalid_nets.insert(net);
}

}  // namespace ito