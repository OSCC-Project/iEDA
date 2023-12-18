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

#include "TimingAnnotation.hh"

#include <algorithm>

#include "utility/Utility.hh"

namespace ipl {
//

void TimingAnnotation::init()
{
  // TODO init steiner wirelength and init all topology order
}

Node* TimingAnnotation::get_clock_node(Group* group)
{
  Node* clock_node = nullptr;
  for (auto* node : group->obtainInputNodes()) {
    auto* network = node->get_network();
    if (network) {
      if (network->get_network_type() == NETWORK_TYPE::kClock) {
        clock_node = node;
        break;
      }
    }
  }

  return clock_node;
}

Node* TimingAnnotation::get_data_node(Group* group)
{
  Node* data_node = nullptr;
  for (auto* node : group->obtainInputNodes()) {
    auto* network = node->get_network();
    if (network) {
      if (network->get_network_type() != NETWORK_TYPE::kClock) {
        data_node = node;
        break;
      }
    }
  }

  return data_node;
}

float TimingAnnotation::get_node_criticality(Node* node)
{
  float node_criticality = 0.0f;
  float node_slack = get_node_late_slack(node->get_node_id());
  node_slack > 0 ? node_slack = 0 : node_slack;
  float wns = get_late_wns();

  if (wns > 0) {
    return 0.0f;
  } else {
    return std::max(0.0f, std::min((float) (1.0), node_slack / wns));
  }
}

float TimingAnnotation::get_group_criticality(Group* group)
{
  float criticality = 0.0f;
  for (auto* node : group->obtainOutputNodes()) {
    criticality = std::max(criticality, get_node_criticality(node));
  }

  if (group->get_group_type() == GROUP_TYPE::kFlipflop) {
    // obtain the dpin of flipflop
  }
}

void TimingAnnotation::updateCriticalityAndCentralityFull()
{
  // reset max centrality.
  _max_centrality = 0.0f;

  // update all network in reverse order
  int32_t network_size = _topo_order_net_list.size();
  for (int32_t i = network_size - 1; i >= 0; i--) {
    updateCriticalityAndCentrality(_topo_order_net_list[i]);
  }
}

void TimingAnnotation::updateCriticalityAndCentrality(NetWork* network)
{
  bool dont_propagate_clock_net = true;

  float sum_sink_centrality = 0.0f;
  for (auto* sink : network->get_receiver_list()) {
    sink->set_centrality(0.0);

    // update the centrality of this sink.
    bool has_arcs = false;
    for (auto* arc : sink->get_output_arc_list()) {
      auto* to_node = arc->get_to_node();

      has_arcs = true;
      double centrality = sink->get_centrality() + arc->get_flow_value() * to_node->get_centrality();
      sink->set_centrality(centrality);
    }
    // if no arcs were found, keep set its centrality to its criticality.
    if (!has_arcs) {
      double centrality = get_node_criticality(sink);
      sink->set_centrality(centrality);
    }

    bool is_clock_pin = (sink->get_network()->get_network_type() == NETWORK_TYPE::kClock);
    if (!dont_propagate_clock_net || !is_clock_pin) {
      sum_sink_centrality += sink->get_centrality();
    }
  }

  // update maximum centrality
  _max_centrality = std::max(_max_centrality, sum_sink_centrality);

  // update driver centrality and driving arc flows.
  // assumption is that driver net is single.
  auto* node_driver = network->get_transmitter();
  if (node_driver) {
    node_driver->set_centrality(sum_sink_centrality);
  }

  // update driving arcs flow
  float sum = 0.0f;
  int counter_arcs = 0;
  for (auto* arc : node_driver->get_input_arc_list()) {
    auto* from_node = arc->get_from_node();
    sum += get_node_criticality(from_node);
    counter_arcs++;
  }

  if (Utility().isFloatApproximatelyZero(sum)) {
    sum = 1 / float(counter_arcs);
  }

  for (auto* arc : node_driver->get_input_arc_list()) {
    auto* from_node = arc->get_from_node();
    float flow_value = get_node_criticality(from_node) / sum;
    arc->set_flow_value(flow_value);
  }
}

}  // namespace ipl