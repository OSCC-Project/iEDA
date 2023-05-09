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
#include "MinStepCheck.hpp"

#include "DRCUtil.h"

namespace idrc {

// operation api
bool MinStepCheck::check(DrcNet* target_net)
{
  checkMinStep(target_net);
  return _check_result;
}

bool MinStepCheck::check(DrcPoly* target_poly)
{
  checkMinStep(target_poly);
  return _check_result;
}

void MinStepCheck::checkMinStep(DrcNet* target_net)
{
  for (auto& [layer_id, target_polys] : target_net->get_route_polys_list()) {
    for (auto& target_poly : target_polys) {
      checkMinStep(target_poly.get());
    }
  }
}

void MinStepCheck::checkMinStep(DrcPoly* target_poly)
{
  int layer_id = target_poly->get_layer_id();
  _lef58_rule = _tech->get_drc_routing_layer_list()[layer_id]->get_lef58_min_step_rule();
  _rule = _tech->get_drc_routing_layer_list()[layer_id]->get_min_step_rule();

  // if layer dont have notch rule,skip check
  if ((_lef58_rule.size() > 0) && (_rule == nullptr)) {
    return;
  }
  int size = _lef58_rule.size();
  for (_rule_index = 0; _rule_index < size; _rule_index++) {
    // A polygon with holes has multiple sets of edges
    for (auto& edges : target_poly->getEdges()) {
      for (auto& edge : edges) {
        if (isTriggerEdge(edge.get()) && _vio_edge_set.find(edge.get()) == _vio_edge_set.end()) {
          checkMinStep(edge.get());
        }
      }
    }
    for (auto& edges : target_poly->getEdges()) {
      for (auto& edge : edges) {
        if (isTriggerEdge_lef58(edge.get())) {
          checkMinStep_lef58(edge.get());
        }
      }
    }
  }
}

bool MinStepCheck::isTriggerEdge_lef58(DrcEdge* edge)
{
  return edge->getLength() < _lef58_rule[_rule_index]->get_min_step_length();
}

void MinStepCheck::checkAdjEdgeLength(DrcEdge* edge, DrcEdge* adj_edge)
{
  if (adj_edge->getLength() < _lef58_rule[_rule_index]->get_min_adjacent_length()->get_min_adj_length()) {
    if (_interact_with_op) {
      _region_query->addViolation(ViolationType::kMinStep);
      addSpot_lef58(edge, adj_edge);
      _check_result = false;
    } else {
      // TODO
      // addspot();
    }
  }
}

void MinStepCheck::checkNextEdge_lef58(DrcEdge* edge)
{
  auto pre_edge = edge->getPreEdge();
  auto next_edge = edge->getNextEdge();
  if (DRCUtil::isCornerConVex(edge) && DRCUtil::isCornerConCave(pre_edge) && DRCUtil::isCornerConCave(next_edge)) {
    checkAdjEdgeLength(edge, next_edge);
  }
}

void MinStepCheck::checkPrevEdge_lef58(DrcEdge* edge)
{
  auto pre_edge = edge->getPreEdge();
  // auto next_edge = edge->getNextEdge();
  if (DRCUtil::isCornerConVex(pre_edge) && DRCUtil::isCornerConCave(edge) && DRCUtil::isCornerConCave(pre_edge->getPreEdge())) {
    checkAdjEdgeLength(edge, pre_edge);
  }
}

void MinStepCheck::checkMinStep_lef58(DrcEdge* edge)
{
  checkNextEdge_lef58(edge);
  checkPrevEdge_lef58(edge);
}

void MinStepCheck::checkMinStep(DrcEdge* edge)
{
  int count = 1;
  refresh(edge);
  while (isTriggerEdge(edge->getNextEdge()) && edge->getNextEdge() != _trigger_edge) {
    edge = edge->getNextEdge();
    _end_edge = edge;
    count++;
  }
  if (edge->getNextEdge() != _trigger_edge) {
    edge = _trigger_edge;
    while (isTriggerEdge(edge->getPreEdge())) {
      edge = edge->getPreEdge();
      _begin_edge = edge;
      count++;
    }
  }

  if (count > _rule->get_max_edges()) {
    if (checkCorner()) {
      if (_interact_with_op) {
        _region_query->addViolation(ViolationType::kMinStep);
        addSpot(_begin_edge, _end_edge);

        _check_result = false;
      } else {
        // TODO:
        std::cout << _trigger_edge->get_begin_x() << "," << _trigger_edge->get_begin_y() << " Vio!!!!!" << std::endl;
        //   addspot();
      }
    }
  }
}

void MinStepCheck::addSpot_lef58(DrcEdge* edge, DrcEdge* adj_edge)
{
  int lb_x, lb_y, rt_x, rt_y;
  lb_x = std::min(adj_edge->get_min_x(), edge->get_min_x());
  lb_y = std::min(adj_edge->get_min_y(), edge->get_min_y());
  rt_x = std::max(adj_edge->get_max_x(), edge->get_max_x());
  rt_y = std::max(adj_edge->get_max_y(), edge->get_max_y());
  DrcViolationSpot* spot = new DrcViolationSpot();
  int layer_id = edge->get_layer_id();
  spot->set_layer_id(layer_id);
  spot->set_layer_name(_tech->getRoutingLayerNameById(layer_id));
  // spot->set_net_id(edge->getNetId());
  spot->set_vio_type(ViolationType::kMinStep);
  spot->setCoordinate(lb_x, lb_y, rt_x, rt_y);
  _region_query->_min_step_spot_list.emplace_back(spot);
}
void MinStepCheck::addSpot(DrcEdge* begin_edge, DrcEdge* end_edge)
{
  //
  _vio_edge_set.clear();
  int lb_x = 1e9, lb_y = 1e9, rt_x = 0, rt_y = 0;
  auto edge = begin_edge;
  while (edge != end_edge) {
    lb_x = std::min(lb_x, edge->get_min_x());
    lb_y = std::min(lb_y, edge->get_min_y());
    rt_x = std::max(rt_x, edge->get_max_x());
    rt_y = std::max(rt_y, edge->get_max_y());
    _vio_edge_set.insert(edge);
    edge = edge->getNextEdge();
  }
  lb_x = std::min(lb_x, end_edge->get_min_x());
  lb_y = std::min(lb_y, end_edge->get_min_y());
  rt_x = std::max(rt_x, end_edge->get_max_x());
  rt_y = std::max(rt_y, end_edge->get_max_y());
  _vio_edge_set.insert(end_edge);
  DrcViolationSpot* spot = new DrcViolationSpot();
  int layer_id = edge->get_layer_id();
  spot->set_layer_id(layer_id);
  spot->set_layer_name(_tech->getRoutingLayerNameById(layer_id));
  // spot->set_net_id(edge->getNetId());
  spot->set_vio_type(ViolationType::kMinStep);

  // if (edge->isHorizontal()) {
  //   lb_x = edge->get_min_x();
  //   rt_x = edge->get_max_x();
  //   lb_y = std::min();
  //   rt_y = std::min();
  // }
  // if (edge->isHorizontal()) {
  //   lb_x = edge->get_min_x();
  //   rt_x = edge->get_max_x();
  //   lb_y = edge->get_min_y();
  //   rt_y = edge->get_min_y() + 1;
  // } else {
  //   lb_x = edge->get_min_x();
  //   rt_x = edge->get_max_x() + 1;
  //   lb_y = edge->get_min_y();
  //   rt_y = edge->get_max_y();
  // }
  // spot->setCoordinate(edge->get_min_x(), edge->get_min_y(), edge->get_max_x(), edge->get_max_y());
  spot->setCoordinate(lb_x, lb_y, rt_x, rt_y);
  _region_query->_min_step_spot_list.emplace_back(spot);
}

void MinStepCheck::refresh(DrcEdge* edge)
{
  _trigger_edge = edge;
  _begin_edge = edge;
  _end_edge = edge;
}

bool MinStepCheck::checkBeginCorner()
{
  return DRCUtil::isCornerConVex(_end_edge);
}

bool MinStepCheck::checkEndCorner()
{
  return DRCUtil::isCornerConVex(_begin_edge->getPreEdge());
}

bool MinStepCheck::checkCorner()
{
  return checkBeginCorner() && checkEndCorner() && !(DRCUtil::isTwoEdgeParallel(_end_edge->getNextEdge(), _begin_edge->getPreEdge()));
}

bool MinStepCheck::isTriggerEdge(DrcEdge* edge)
{
  return edge->getLength() < _rule->get_min_step_length();
}

}  // namespace idrc