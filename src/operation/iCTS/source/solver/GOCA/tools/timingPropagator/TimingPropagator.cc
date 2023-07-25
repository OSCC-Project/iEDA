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
/**
 * @file TimingPropagator.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "TimingPropagator.hh"

#include <ranges>

#include "CTSAPI.hpp"
#include "CtsConfig.h"
namespace icts {
double TimingPropagator::_unit_cap = 0;
double TimingPropagator::_unit_res = 0;
double TimingPropagator::_skew_bound = 0;
int TimingPropagator::_db_unit = 0;
double TimingPropagator::_max_buf_tran = 0;
double TimingPropagator::_max_sink_tran = 0;
double TimingPropagator::_max_cap = 0;
int TimingPropagator::_max_fanout = 0;
double TimingPropagator::_max_length = 0;
double TimingPropagator::_min_insert_delay = 0;
std::vector<icts::CtsCellLib*> TimingPropagator::_delay_libs;

/**
 * @brief init timing parameters
 *       this function should be called before any other function
 *
 */
void TimingPropagator::init()
{
  // set RC, db unit and liberty info from api
  _unit_res = CTSAPIInst.getClockUnitRes() / 1000;  
  _unit_cap = CTSAPIInst.getClockUnitCap();         
  _db_unit = CTSAPIInst.getDbUnit();
  _delay_libs = CTSAPIInst.getAllBufferLibs();
  // set algorithm parameters from config
  auto* config = CTSAPIInst.get_config();
  _skew_bound = config->get_skew_bound();
  _max_buf_tran = config->get_max_buf_tran();
  _max_sink_tran = config->get_max_sink_tran();
  _max_cap = config->get_max_cap();
  _max_fanout = config->get_max_fanout();
  _max_length = config->get_max_length();
  // temp para
  _min_insert_delay = _delay_libs.front()->getDelayIntercept();
}
/**
 * @brief update all timing information
 *       1. propagate fanout and cap by postorder
 *       2. propagate slew by preorder
 *       3. propagate delay by preorder
 *
 * @param node
 * @param propagete_type
 */
void TimingPropagator::update(Node* node, const PropagateType& propagete_type)
{
  fanoutPropagate(node, propagete_type);
  capPropagate(node, propagete_type);
  slewPropagate(node, propagete_type);
  delayPropagate(node, propagete_type);
}
/**
 * @brief propagate fanout by post order
 *
 * @param node
 * @param propagete_type
 */
void TimingPropagator::fanoutPropagate(Node* node, const PropagateType& propagete_type)
{
  switch (propagete_type) {
    case PropagateType::kNET:
      node->postOrderBy(updateFanout);
      break;
    case PropagateType::kALL:
      node->postOrder(updateFanout);
      break;
    default:
      break;
  }
}
/**
 * @brief propagate cap by post order
 *
 * @param node
 * @param propagete_type
 */
void TimingPropagator::capPropagate(Node* node, const PropagateType& propagete_type)
{
  switch (propagete_type) {
    case PropagateType::kNET:
      node->postOrderBy(updateCap);
      break;
    case PropagateType::kALL:
      node->postOrder(updateCap);
      break;
    default:
      break;
  }
}
/**
 * @brief propagate slew by pre order
 *
 * @param node
 * @param propagete_type
 */
void TimingPropagator::slewPropagate(Node* node, const PropagateType& propagete_type)
{
  switch (propagete_type) {
    case PropagateType::kNET:
      node->preOrderBy(updateSlew);
      break;
    case PropagateType::kALL:
      node->preOrder(updateSlew);
      break;
    default:
      break;
  }
}
/**
 * @brief propagate delay by post order
 *
 * @param node
 * @param propagete_type
 */
void TimingPropagator::delayPropagate(Node* node, const PropagateType& propagete_type)
{
  switch (propagete_type) {
    case PropagateType::kNET:
      node->postOrderBy(updateDelay);
      break;
    case PropagateType::kALL:
      node->postOrder(updateDelay);
      break;
    default:
      break;
  }
}
/**
 * @brief update fanout
 *       buffer node or sink node: fanout is set to 1 (default value)
 *       steiner node: fanout is set by sum of children's fanout
 *
 * @param node
 */
void TimingPropagator::updateFanout(Node* node)
{
  if (node->isSteiner()) {
    auto fanout = calcFanout(node);
    node->set_fanout(fanout);
  } else {
    node->set_fanout(1);
  }
}
/**
 * @brief update pin's cap load and cap out
 *       cap_load: cap of load pin
 *       cap_out: cap of driver pin
 *
 * @param node
 */
void TimingPropagator::updateCap(Node* node)
{
  if (!node->isSteiner()) {
    auto cell_name = node->get_cell_master();
    auto* lib = CTSAPIInst.getCellLib(cell_name);
    node->set_cap_load(lib->get_init_cap());
  }
  auto cap_out = calcCap(node);
  node->set_cap_out(cap_out);
}
/**
 * @brief update slew for [child nodes], for slew constraint and insertion delay calculation
 *       buffer node: slew_out is calculated by cap_out in buffer lib (Linear Characteristics)
 *                    child node's slew_in is calculated by slew_out and slew_ideal
 *       steiner node: child node's slew_in is calculated by parent's slew_in and slew_ideal
 *
 * @param node
 */
void TimingPropagator::updateSlew(Node* node)
{
  auto calc_slew_in = [&node](Node* child) {
    auto slew_ideal = calcIdealSlew(node, child);
    double slew_in = 0;
    if (node->isBuffer()) {
      double slew_out = 0;
      if (node->isBuffer()) {
        auto cell_name = node->get_cell_master();
        auto* lib = CTSAPIInst.getCellLib(cell_name);
        slew_out = lib->calcSlew(node->get_cap_out());
      }
      slew_in = std::sqrt(std::pow(slew_out, 2) + std::pow(slew_ideal, 2));
    } else {
      slew_in = std::sqrt(std::pow(node->get_slew_in(), 2) + std::pow(slew_ideal, 2));
    }
    child->set_slew_in(slew_in);
  };
  std::ranges::for_each(node->get_children(), calc_slew_in);
}
/**
 * @brief update node's delay and child node's insertion delay
 *       wire delay: elmore model delay
 *       buffer insertion delay: delay of buffer lib, calculated by slew_in and cap_out
 *
 * @param node
 */
void TimingPropagator::updateDelay(Node* node)
{
  double min_delay = std::numeric_limits<double>::max();
  double max_delay = std::numeric_limits<double>::min();
  auto calc_delay = [&min_delay, &max_delay, &node](Node* child) {
    auto delay = calcElmoreDelay(node, child);
    if (child->isBuffer()) {
      auto cell_name = child->get_cell_master();
      auto* lib = CTSAPIInst.getCellLib(cell_name);
      auto insert_delay = lib->calcDelay(child->get_slew_in(), child->get_cap_out());
      child->set_insert_delay(insert_delay);
      delay += insert_delay;
    }
    min_delay = std::min(min_delay, delay + node->get_min_delay());
    max_delay = std::max(max_delay, delay + node->get_max_delay());
  };
  std::ranges::for_each(node->get_children(), calc_delay);
  node->set_min_delay(min_delay);
  node->set_max_delay(max_delay);
}
/**
 * @brief calculate fanout
 *       if child is buffer or sink, parent's fanout is added by 1,
 *          else parent's fanout is added by child's fanout
 *
 * @param node
 * @return uint16_t
 */
uint16_t TimingPropagator::calcFanout(Node* node)
{
  uint16_t fanout = 0;
  auto accumulate_fanout = [&fanout](Node* child) { fanout += child->isSteiner() ? child->get_fanout() : 1; };
  std::ranges::for_each(node->get_children(), accumulate_fanout);
  return fanout;
}
/**
 * @brief calculate capacitance
 *       if child is steiner node, parent's cap_out is added by unit_cap * len + child's cap_out,
 *          else parent's cap_out is added by unit_cap * len + child's cap_load (pin cap)
 *
 * @param node
 * @return double
 */
double TimingPropagator::calcCap(Node* node)
{
  double cap_out = 0;
  auto accumulate_cap = [&cap_out, &node](Node* child) {
    auto sub_cap = child->isSteiner() ? child->get_cap_out() : child->get_cap_load();
    cap_out += _unit_cap * calcLen(node, child) + sub_cap;
  };
  std::ranges::for_each(node->get_children(), accumulate_cap);
  return cap_out;
}
/**
 * @brief calculate ideal slew (wire slew)
 *       ideal_slew = log(9) * elmore_delay
 *
 * @param node
 * @param child
 * @return double
 */
double TimingPropagator::calcIdealSlew(Node* node, Node* child)
{
  return std::log(9) * calcElmoreDelay(node, child);
}
/**
 * @brief calculate elmore delay
 *       elmore_delay = r * l * (c * l / 2 + c_t)
 *
 * @param parent
 * @param child
 * @return double
 */
double TimingPropagator::calcElmoreDelay(Node* parent, Node* child)
{
  auto len = calcLen(parent, child);
  auto delay = _unit_res * len * (_unit_cap * len / 2 + child->get_cap_out());
  return delay;
}
/**
 * @brief calculate manhattan wire length between n1 and n2
 *       l = D(n1,n2) / db_unit
 *
 * @param n1
 * @param n2
 * @return double
 */
double TimingPropagator::calcLen(Node* n1, Node* n2)
{
  return 1.0 * calcDist(n1, n2) / _db_unit;
}
/**
 * @brief calculate manhattan dist between n1 and n2
 *       D(n1,n2) = |x1-x2| + |y1-y2|
 *
 * @param n1
 * @param n2
 * @return int64_t
 */
int64_t TimingPropagator::calcDist(Node* n1, Node* n2)
{
  auto p1 = n1->get_location();
  auto p2 = n2->get_location();
  return std::abs(p1.x() - p2.x()) + std::abs(p1.y() - p2.y());
}
}  // namespace icts