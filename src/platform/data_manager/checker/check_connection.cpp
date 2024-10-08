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
#include "check_connection.h"

#include "../idm.h"
#include "IdbNet.h"
#include "IdbPins.h"
#include "IdbRegularWire.h"

namespace idm {

bool CheckNodePin::isIntersection(CheckNode* node_dst)
{
  if (node_dst->is_seg()) {
    CheckNodeSegment* seg_node = dynamic_cast<CheckNodeSegment*>(node_dst);

    auto segment = seg_node->get_segment();

    /// only segment of routing layer can be connected in pin
    for (auto layer_shape : _pin->get_port_box_list()) {
      if (segment->isIntersection(layer_shape)) {
        return true;
      }
    }
  }

  return false;
}

bool CheckNodeSegment::isIntersection(CheckNode* node_dst)
{
  if (node_dst->is_seg()) {
    CheckNodeSegment* node_seg_dst = dynamic_cast<CheckNodeSegment*>(node_dst);

    return _segment->isIntersection(node_seg_dst->get_segment());
  }

  return false;
}

void CheckNet::wrapNet(IdbNet* net)
{
  _net_name = net->get_net_name();
  _pin_num = net->get_pin_number();
  _node_list.reserve(_pin_num + net->get_segment_num());
  wrapNetPinList(net);
  wrapNetSegmentList(net);
}

int CheckNet::wrapNetPinList(IdbNet* net)
{
  /// instance pin
  for (auto pin : net->get_instance_pin_list()->get_pin_list()) {
    CheckNodePin* node = new CheckNodePin(pin, _node_list.size());

    _node_list.push_back(node);
  }

  /// IO pin
  auto* io_pins = net->get_io_pins();
  for (auto* io_pin : io_pins->get_pin_list()) {
    CheckNodePin* node = new CheckNodePin(io_pin, _node_list.size());

    _node_list.push_back(node);
  }

  return _node_list.size();
}

int CheckNet::wrapNetSegmentList(IdbNet* net)
{
  auto wire_list = net->get_wire_list();
  if (wire_list != nullptr) {
    for (auto wire : wire_list->get_wire_list()) {
      for (auto seg : wire->get_segment_list()) {
        CheckNodeSegment* node = new CheckNodeSegment(seg, _node_list.size());

        _node_list.push_back(node);
      }
    }
  }

  return _node_list.size();
}

void CheckNet::buildGraph()
{
  int index = 0;
  for (size_t i = 0; i < _node_list.size(); i++) {
    if (_node_list[i]->is_visited()) {
      continue;
    }

    NetGraph graph;
    graph.set_id(index);

    /// add vertex to graph
    _node_list[i]->set_graph_id(index);
    graph.add_vertex(i);

    /// add pin to graph
    if (_node_list[i]->is_pin()) {
      graph.addConnectedPin(i);
    }

    buildGraphBFS(graph, _node_list[i]);

    _graph_list.push_back(graph);

    index++;

    // std::cout << "[CheckNet Info] Net = " << _net_name << " graph id = " << graph.get_id() << " vertex_num = " << graph.get_vertex_num()
    //           << " edge_num = " << graph.get_edge_num() << " pin_num = " << graph.get_pin_num() << std::endl;
  }
}

void CheckNet::buildGraphBFS(NetGraph& graph, CheckNode* check_node)
{
  std::vector<CheckNode*> connected_node_list;
  /// find connected vetex
  for (size_t i = 0; i < _node_list.size(); i++) {
    if (_node_list[i]->is_visited() || check_node == _node_list[i]) {
      continue;
    }

    if (isIntersection(check_node, _node_list[i])) {
      /// set as visted
      _node_list[i]->set_graph_id(graph.get_id());
      check_node->set_graph_id(graph.get_id());

      /// add to graph
      graph.add_vertex(i);
      graph.add_edge(check_node->get_id(), _node_list[i]->get_id());

      /// add pin to graph
      if (_node_list[i]->is_pin()) {
        graph.addConnectedPin(i);
      }

      connected_node_list.push_back(_node_list[i]);
    }
  }

  /// BFS to process connected node
  for (auto connect_node : connected_node_list) {
    buildGraphBFS(graph, connect_node);
  }
}

std::vector<CheckNode*> CheckNet::get_node_list(NodeType type)
{
  std::vector<CheckNode*> node_list;
  for (auto pin_node : _node_list) {
    if (pin_node->get_type() == type) {
      node_list.push_back(pin_node);
    }
  }

  return node_list;
}

CheckInfo CheckNet::checkNetConnection()
{
  if (get_pin_num() < 2 || get_segment_num() <= 0) {
    return CheckInfo::kDisconnected;
  }

  buildGraph();

  hasRing();

  return isAllPinConnected();
}

CheckInfo CheckNet::isAllPinConnected()
{
  for (auto net_graph : _graph_list) {
    if (_pin_num >= 0 && _pin_num == net_graph.get_pin_num()) {
      // std::cout << "[CheckNet Info] Net " << _net_name << " is connected." << std::endl;
      return CheckInfo::kConnected;
    }
  }

  std::cout << "[CheckNet Error] Net " << _net_name << " is disconnected." << std::endl;
  return CheckInfo::kDisconnected;
}

bool CheckNet::hasRing()
{
  for (auto net_graph : _graph_list) {
    if (net_graph.has_ring()) {
      std::cout << "[CheckNet Error] Net " << _net_name << " has ring."
                << " vertex_num = " << net_graph.get_vertex_num() << " edge_num = " << net_graph.get_edge_num() << std::endl;
      return true;
    }
  }

  //   std::cout << "[CheckNet Info] Net " << _net_name << " has no ring." << std::endl;
  return false;
}

/**
 * @brief check node intersection, at least one of the two node must be segment
 *
 * @param node_src
 * @param node_dst
 * @return true
 * @return false
 */
bool CheckNet::isIntersection(CheckNode* node_src, CheckNode* node_dst)
{
  /// 2 pin node can not be connected
  if (node_src->is_pin() && node_dst->is_pin()) {
    return false;
  }

  /// 1 pin, 1 segment
  if (node_src->is_pin()) {
    CheckNodePin* node_pin = dynamic_cast<CheckNodePin*>(node_src);
    CheckNodeSegment* node_seg = dynamic_cast<CheckNodeSegment*>(node_dst);
    return node_pin->isIntersection(node_seg);
  }
  /// 1 pin, 1 segment
  if (node_dst->is_pin()) {
    CheckNodePin* node_pin = dynamic_cast<CheckNodePin*>(node_dst);
    CheckNodeSegment* node_seg = dynamic_cast<CheckNodeSegment*>(node_src);
    return node_pin->isIntersection(node_seg);
  }

  /// 2 segment
  return (dynamic_cast<CheckNodeSegment*>(node_src))->isIntersection(dynamic_cast<CheckNodeSegment*>(node_dst));
}

}  // namespace idm