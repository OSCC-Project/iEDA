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
#include "TreeBuilder.hh"
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
  _unit_cap = CTSAPIInst.getClockUnitCap();
  _unit_res = CTSAPIInst.getClockUnitRes() / 1000;
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
 * @brief generate net from driver pin and load pins
 *
 * @param net_name
 * @param driver_pin
 * @param load_pins
 * @return Net*
 */
Net* TimingPropagator::genNet(const std::string& net_name, Pin* driver_pin, const std::vector<Pin*>& load_pins)
{
  if (load_pins.empty()) {
    std::vector<Pin*> loads;
    driver_pin->preOrder([&loads](Node* node) {
      if (node->isPin() && node->isLoad()) {
        loads.push_back(dynamic_cast<Pin*>(node));
      }
    });
    return new Net(net_name, driver_pin, loads);
  } else {
    return new Net(net_name, driver_pin, load_pins);
  }
}
/**
 * @brief update net's load pins
 *
 * @param net
 */
void TimingPropagator::updateLoads(Net* net)
{
  auto* driver_pin = net->get_driver_pin();
  std::vector<Pin*> loads;
  driver_pin->preOrder([&loads](Node* node) {
    if (node->isPin() && node->isLoad()) {
      loads.push_back(dynamic_cast<Pin*>(node));
    }
  });
  net->set_load_pins(loads);
}
/**
 * @brief update all timing information
 *       propagate net_len, cap, slew, cell_delay, wire_delay
 *
 * @param net
 */
void TimingPropagator::update(Net* net)
{
  netLenPropagate(net);
  capPropagate(net);
  slewPropagate(net);
  cellDelayPropagate(net);
  wireDelayPropagate(net);
}
/**
 * @brief propagate net's wirelength by post order
 *
 * @param net
 */
void TimingPropagator::netLenPropagate(Net* net)
{
  auto* driver_pin = net->get_driver_pin();
  driver_pin->postOrder(updateNetLen);
}
/**
 * @brief propagate cap by post order, and update load pin's cap
 *
 * @param net
 */
void TimingPropagator::capPropagate(Net* net)
{
  auto* driver_pin = net->get_driver_pin();
  driver_pin->postOrder(updateCapLoad);
}
/**
 * @brief propagate slew by pre order
 *
 * @param net
 */
void TimingPropagator::slewPropagate(Net* net)
{
  auto* driver_pin = net->get_driver_pin();
  driver_pin->preOrder(updateSlewIn);
}
/**
 * @brief propagate cell delay
 *
 * @param net
 */
void TimingPropagator::cellDelayPropagate(Net* net)
{
  std::ranges::for_each(net->get_load_pins(), [](Pin* pin) {
    auto* inst = pin->get_inst();
    updateCellDelay(inst);
  });
}
/**
 * @brief propagate wire delay by post order
 *
 * @param net
 */
void TimingPropagator::wireDelayPropagate(Net* net)
{
  auto* driver_pin = net->get_driver_pin();
  driver_pin->postOrder(updateWireDelay);
}
/**
 * @brief update insertion delay
 *       insert_delay = a * [slew_in] + b * [cap_load] + c, the function is from liberty fitting
 *
 * @param inst
 */
void TimingPropagator::updateCellDelay(Inst* inst)
{
  if (!inst->isBuffer()) {
    return;
  }
  auto* driver_pin = inst->get_driver_pin();
  auto* load_pin = inst->get_load_pin();
  auto slew_in = load_pin->get_slew_in();
  auto cap_load = driver_pin->get_cap_load();
  auto cell_name = inst->get_cell_master();
  auto* lib = CTSAPIInst.getCellLib(cell_name);
  auto insert_delay = lib->calcDelay(slew_in, cap_load);
  inst->set_insert_delay(insert_delay);
  auto min_delay = driver_pin->get_min_delay();
  auto max_delay = driver_pin->get_max_delay();
  load_pin->set_min_delay(insert_delay + min_delay);
  load_pin->set_max_delay(insert_delay + max_delay);
}
/**
 * @brief update net's wirelength
 *
 * @param node
 */
void TimingPropagator::updateNetLen(Node* node)
{
  auto net_len = calcNetLen(node);
  node->set_sub_len(net_len);
}
/**
 * @brief update pin's cap load and cap out
 *
 * @param node
 */
void TimingPropagator::updateCapLoad(Node* node)
{
  auto cap_load = calcCapLoad(node);
  node->set_cap_load(cap_load);
}
/**
 * @brief update slew, for slew constraint and insertion delay calculation
 *       driver pin: slew_out = a * [cap_load] + b, the function is from liberty fitting
 *       load pin or steiner node: slew_out = [slew_in]
 *                                 slew_in = sqrt([parent's slew_out]^2 + [slew_wire]^2)
 *
 * @param node
 */
void TimingPropagator::updateSlewIn(Node* node)
{
  auto calc_slew_in = [&node](Node* child) {
    auto slew_ideal = calcIdealSlew(node, child);
    double slew_in = 0;
    if (node->isBufferPin() && node->isDriver()) {
      auto cell_name = node->getCellMaster();
      auto* lib = CTSAPIInst.getCellLib(cell_name);
      auto slew_out = lib->calcSlew(node->get_cap_load());
      slew_in = std::sqrt(std::pow(slew_out, 2) + std::pow(slew_ideal, 2));
    } else {
      slew_in = std::sqrt(std::pow(node->get_slew_in(), 2) + std::pow(slew_ideal, 2));
    }
    child->set_slew_in(slew_in);
  };
  std::ranges::for_each(node->get_children(), calc_slew_in);
}
/**
 * @brief update wire delay between node and its child
 *       wire delay: elmore model delay
 *
 * @param node
 */
void TimingPropagator::updateWireDelay(Node* node)
{
  if (node->get_children().empty()) {
    return;
  }
  double min_delay = std::numeric_limits<double>::max();
  double max_delay = std::numeric_limits<double>::min();
  auto calc_delay = [&min_delay, &max_delay, &node](Node* child) {
    auto delay = calcElmoreDelay(node, child);
    min_delay = std::min(min_delay, delay + child->get_min_delay());
    max_delay = std::max(max_delay, delay + child->get_max_delay());
  };
  std::ranges::for_each(node->get_children(), calc_delay);
  if (node->isPin()) {
    auto* pin = dynamic_cast<Pin*>(node);
    if (pin->isLoad()) {
      min_delay = std::min(min_delay, pin->get_min_delay());
      max_delay = std::max(max_delay, pin->get_max_delay());
    }
  }
  node->set_min_delay(min_delay);
  node->set_max_delay(max_delay);
}
/**
 * @brief calculate downstream wirelength
 *
 * @param node
 * @return double
 */
double TimingPropagator::calcNetLen(Node* node)
{
  double net_len = 0;
  auto accumulate_net_len = [&net_len, &node](Node* child) { net_len += calcLen(node, child) + child->get_sub_len(); };
  std::ranges::for_each(node->get_children(), accumulate_net_len);
  return net_len;
}
/**
 * @brief calculate capacitance
 *
 * @param node
 * @return double
 */
double TimingPropagator::calcCapLoad(Node* node)
{
  double cap_load = 0;
  if (node->isSinkPin() && node->isLoad()) {
    auto* pin = dynamic_cast<Pin*>(node);
    auto* inst = pin->get_inst();
    auto* cts_inst = inst->get_cts_inst();
    cap_load = CTSAPIInst.getSinkCap(cts_inst);
  }
  if (node->isBufferPin() && node->isLoad()) {
    auto cell_name = node->getCellMaster();
    auto* lib = CTSAPIInst.getCellLib(cell_name);
    cap_load = lib->get_init_cap();
  }
  auto accumulate_cap = [&cap_load, &node](Node* child) { cap_load += _unit_cap * calcLen(node, child) + child->get_cap_load(); };
  std::ranges::for_each(node->get_children(), accumulate_cap);
  return cap_load;
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
  auto delay = _unit_res * len * (_unit_cap * len / 2 + child->get_cap_load());
  return delay;
}
/**
 * @brief calculate wirelength between parent and child (consider snake)
 *
 * @param parent
 * @param child
 * @return double
 */
double TimingPropagator::calcLen(Node* parent, Node* child)
{
  auto parent_loc = parent->get_location();
  auto child_loc = child->get_location();
  auto len = calcLen(parent_loc, child_loc);
  if (child->get_parent() == parent) {
    len += child->get_required_snake();
  }
  return len;
}
/**
 * @brief calculate manhattan dist between parent and child (consider snake)
 *
 * @param parent
 * @param child
 * @return int64_t
 */
int64_t TimingPropagator::calcDist(Node* parent, Node* child)
{
  auto parent_loc = parent->get_location();
  auto child_loc = child->get_location();
  auto dist = calcDist(parent_loc, child_loc);
  if (child->get_parent() == parent) {
    dist += child->get_required_snake() + static_cast<int64_t>(child->get_required_snake() * _db_unit);
  }
  return dist;
}
/**
 * @brief calculate manhattan wire length between p1 and p2
 *       l = D(n1,n2) / db_unit
 *
 * @param p1
 * @param p2
 * @return double
 */
double TimingPropagator::calcLen(const Point& p1, const Point& p2)
{
  return 1.0 * calcDist(p1, p2) / _db_unit;
}
/**
 * @brief calculate manhattan dist between p1 and p2
 *       D(n1,n2) = |x1 - x2| + |y1 - y2|
 *
 * @param p1
 * @param p2
 * @return int64_t
 */
int64_t TimingPropagator::calcDist(const Point& p1, const Point& p2)
{
  return std::abs(p1.x() - p2.x()) + std::abs(p1.y() - p2.y());
}
/**
 * @brief calculate skew
 *
 * @param node
 * @return double
 */
double TimingPropagator::calcSkew(Node* node)
{
  auto min_delay = node->get_min_delay();
  auto max_delay = node->get_max_delay();
  return max_delay - min_delay;
}
/**
 * @brief check if the skew is feasible
 *
 * @param node
 * @param skew_bound
 * @return true
 * @return false
 */
bool TimingPropagator::skewFeasible(Node* node, const std::optional<double>& skew_bound)
{
  auto skew = calcSkew(node);
  return skew <= skew_bound.value_or(_skew_bound) || (skew - skew_bound.value_or(_skew_bound)) < 1e-6;
}
/**
 * @brief init load pin delay (predict cell delay by cell)
 *
 * @param pin
 * @param by_cell
 */
void TimingPropagator::initLoadPinDelay(Pin* pin, const bool& by_cell)
{
  LOG_FATAL_IF(!pin->isLoad()) << "The pin: " << pin->get_name() << " is not load pin";
  auto* inst = pin->get_inst();
  if (inst->isSink()) {
    pin->set_min_delay(0);
    pin->set_max_delay(0);
  } else {
    auto* driver_pin = inst->get_driver_pin();
    double insert_delay = 0;
    if (by_cell) {
      auto cell_name = pin->getCellMaster();
      for (auto* lib : _delay_libs) {
        if (lib->get_cell_master() == cell_name) {
          auto cap_coef = lib->get_delay_coef().back();
          auto intercept = lib->getDelayIntercept();
          insert_delay = intercept + cap_coef * inst->getCapOut();
        }
      }
    } else {
      auto* max_lib = _delay_libs.back();
      auto cap_coef = max_lib->get_delay_coef().back();
      insert_delay = TimingPropagator::getMinInsertDelay() + cap_coef * inst->getCapOut();
    }
    inst->set_insert_delay(insert_delay);
    pin->set_min_delay(driver_pin->get_min_delay() + insert_delay);
    pin->set_max_delay(driver_pin->get_max_delay() + insert_delay);
  }
}

}  // namespace icts