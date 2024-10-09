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
/*
 * @Author: S.J Chen
 * @Date: 2022-03-09 15:09:04
 * @LastEditTime: 2022-04-06 12:01:32
 * @LastEditors: S.J Chen
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/evaluator/wirelength/HPWirelength.cc
 * Contact : https://github.com/sjchanson
 */

#include "HPWirelength.hh"

#include "omp.h"

namespace ipl {

int64_t HPWirelength::obtainTotalWirelength()
{
  int64_t total_hpwl = 0;

#pragma omp parallel for num_threads(8)
  for (auto* network : _topology_manager->get_network_list()) {
    Rectangle<int32_t> network_shape = std::move(network->obtainNetWorkShape());

#pragma omp atomic
    total_hpwl += network_shape.get_half_perimeter();
  }
  return total_hpwl;
}

std::vector<std::vector<std::pair<int32_t, int32_t>>> HPWirelength::constructPointSets()
{
  std::vector<std::vector<std::pair<int32_t, int32_t>>> point_sets;

  for (auto* network : _topology_manager->get_network_list()) {
    std::vector<std::pair<int32_t, int32_t>> point_set;

    for (auto* node : network->get_node_list()) {
      Point<int32_t> node_loc = node->get_location();
      point_set.emplace_back(node_loc.get_x(), node_loc.get_y());
    }

    point_sets.push_back(point_set);
  }

  return point_sets;
}

int64_t HPWirelength::obtainNetWirelength(int32_t net_id)
{
  int64_t hpwl = 0;

  auto* network = _topology_manager->findNetworkById(net_id);
  if (network) {
    hpwl = network->obtainNetWorkShape().get_half_perimeter();
  }

  return hpwl;
}

int64_t HPWirelength::obtainPartOfNetWirelength(int32_t net_id, int32_t sink_pin_id)
{
  int64_t wirelength = 0;

  auto* network = _topology_manager->findNetworkById(net_id);
  if (network) {
    auto* transmitter = network->get_transmitter();
    if (transmitter) {
      for (auto* receiver : network->get_receiver_list()) {
        if (receiver->get_node_id() == sink_pin_id) {
          wirelength = (std::abs(transmitter->get_location().get_x() - receiver->get_location().get_x())
                        + std::abs(transmitter->get_location().get_y() - receiver->get_location().get_y()));
        }
      }
    }
  }

  return wirelength;
}

}  // namespace ipl