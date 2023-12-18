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

#include "PostGP.hh"

#include <deque>
#include <tuple>

#include "log/Log.hh"
#include "utility/Utility.hh"

namespace ipl {

PostGP::PostGP(Config* pl_config, PlacerDB* placer_db)
{
  _config = pl_config->get_post_gp_config();

  // create database.
  _database = PostGPDatabase();
  _database._placer_db = placer_db;

  // The equality of topo_manager and placer_db should promise by outside.
  auto* topo_manager = placer_db->get_topo_manager();
  _database._inst_list = placer_db->get_design()->get_instance_list();
  _database._net_list = placer_db->get_design()->get_net_list();
  _database._pin_list = placer_db->get_design()->get_pin_list();
  _database._group_list = topo_manager->get_group_copy_list();
  _database._network_list = topo_manager->get_network_copy_list();
  _database._node_list = topo_manager->get_node_copy_list();
}

void PostGP::runBufferBalancing()
{
  LOG_INFO << "Start buffer balancing...";

  // find single buffer chains
  std::deque<std::tuple<float, Instance*>> buffers;

  for (auto* inst : _database._inst_list) {
    auto* group = _database._group_list[inst->get_inst_id()];
    std::vector<Node*> input_nodes = std::move(group->obtainInputNodes());
    std::vector<Node*> output_nodes = std::move(group->obtainOutputNodes());

    if (input_nodes.size() != 1 || output_nodes.size() != 1) {
      continue;
    }

    Node* in_node = input_nodes[0];
    Node* out_node = output_nodes[0];

    NetWork* in_net = in_node->get_network();
    NetWork* out_net = out_node->get_network();

    if (!in_net || in_net->get_node_list().size() != 2) {
      continue;
    }
    if (!out_net || out_net->get_node_list().size() != 2) {
      continue;
    }

    float criticality = _timing_annotation->get_group_criticality(group);
    if (!Utility().isFloatApproximatelyZero(criticality)) {
      buffers.push_back(std::make_tuple(criticality, inst));
    }
  }

  // sort buffer by criticality
  std::sort(buffers.begin(), buffers.end());

  int32_t moved_buffers = 0;
  int32_t failed = 0;

  int num_buffers = buffers.size();

  for (int32_t i = num_buffers - 1; i >= 0; i--) {
    auto* buffer = std::get<1>(buffers[i]);
    if (!doBufferBalancing(buffer)) {
      failed++;
    }
    moved_buffers++;
  }

  LOG_INFO << "End buffer balancing...";
}

bool PostGP::doBufferBalancing(Instance* buffer)
{
  if (buffer->isFixed()) {
    return false;
  }

  //   auto* group = _database._group_list[buffer->get_inst_id()];
  auto inpins = std::move(buffer->get_inpins());
  auto outpins = std::move(buffer->get_outpins());

  Pin* inpin = inpins[0];
  Pin* outpin = outpins[0];

  Net* in_net = inpin->get_net();
  Net* out_net = outpin->get_net();

  Pin* driver = in_net->get_driver_pin();
  if (driver->isIOPort()) {
    return false;
  }

  Pin* sink = out_net->get_sink_pins()[0];

  Node* driver_node = _database._node_list[driver->get_pin_id()];
  Node* out_node = _database._node_list[outpin->get_pin_id()];
  Node* sink_node = _database._node_list[sink->get_pin_id()];
  Node* in_node = _database._node_list[inpin->get_pin_id()];

  float C_w = _timing_annotation->getAvgWireCapPerUnitLength();
  float R_w = _timing_annotation->getAvgWireResPerUnitLength();

  float R_0 = _timing_annotation->getOutNodeRes(driver_node);
  float R_1 = _timing_annotation->getOutNodeRes(out_node);

  float C_1 = _timing_annotation->getNodeInputCap(in_node);
  float C_2 = _timing_annotation->getNodeInputCap(sink_node);

  const Point<int32_t> driver_pos = driver_node->get_location();
  const Point<int32_t> sink_pos = sink_node->get_location();
  const float d = Utility().calManhattanDistance(driver_pos, sink_pos);
  const float a = 0;  // Utility().calManhattanDistance(in_pin_pos, out_pin_pos);

  if (Utility().isFloatApproximatelyZero(d)) {
    return false;
  }

  float d_1
      = std::min(d - a, std::max(0.0f, (C_w * R_1 - C_w * R_0 + R_w * C_2 - R_w * C_1 + (C_w * d - a * C_w) * R_w) / (2 * C_w * R_w)));

  float dx = sink_pos.get_x() - driver_pos.get_x();
  float dy = sink_pos.get_y() - driver_pos.get_y();
  float scaling = d_1 / d;

  float px = scaling * dx + driver_pos.get_x();
  float py = scaling * dy + driver_pos.get_y();

  // TODO calculate the cost and legal the inst.
}

}  // namespace ipl
