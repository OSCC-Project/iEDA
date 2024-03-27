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

#include "CTSAPI.hh"
#include "CtsConfig.hh"
#include "TreeBuilder.hh"
namespace icts {
  double TimingPropagator::_unit_cap = 0;
  double TimingPropagator::_unit_res = 0;
  double TimingPropagator::_unit_h_cap = 0;
  double TimingPropagator::_unit_h_res = 0;
  double TimingPropagator::_unit_v_cap = 0;
  double TimingPropagator::_unit_v_res = 0;
  double TimingPropagator::_skew_bound = 0;
  int TimingPropagator::_db_unit = 0;
  double TimingPropagator::_max_buf_tran = 0;
  double TimingPropagator::_max_sink_tran = 0;
  double TimingPropagator::_max_cap = 0;
  int TimingPropagator::_max_fanout = 0;
  double TimingPropagator::_min_length = 0;
  double TimingPropagator::_max_length = 0;
  double TimingPropagator::_min_insert_delay = 0;
  std::vector<icts::CtsCellLib*> TimingPropagator::_delay_libs;
  icts::CtsCellLib* TimingPropagator::_root_lib = nullptr;

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
    _unit_h_cap = CTSAPIInst.getClockUnitCap(LayerPattern::kH);
    _unit_h_res = CTSAPIInst.getClockUnitRes(LayerPattern::kH) / 1000;
    _unit_v_cap = CTSAPIInst.getClockUnitCap(LayerPattern::kV);
    _unit_v_res = CTSAPIInst.getClockUnitRes(LayerPattern::kV) / 1000;
    _db_unit = CTSAPIInst.getDbUnit();
    _delay_libs = CTSAPIInst.getAllBufferLibs();
    _root_lib = CTSAPIInst.getRootBufferLib();
    // set algorithm parameters from config
    auto* config = CTSAPIInst.get_config();
    _skew_bound = config->get_skew_bound();
    _max_buf_tran = config->get_max_buf_tran();
    _max_sink_tran = config->get_max_sink_tran();
    _max_cap = config->get_max_cap();
    _max_fanout = config->get_max_fanout();
    _min_length = config->get_min_length();
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
    }
    else {
      return new Net(net_name, driver_pin, load_pins);
    }
  }
  /**
   * @brief recover net
   *       remove root node
   *       save the leaf node and disconnect the leaf node
   *
   * @param net
   */
  void TimingPropagator::resetNet(Net* net)
  {
    if (net == nullptr) {
      return;
    }
    auto* driver_pin = net->get_driver_pin();
    auto load_pins = net->get_load_pins();
    std::vector<Node*> to_be_removed;
    auto find_steiner = [&to_be_removed](Node* node) {
      if (node->isSteiner()) {
        to_be_removed.push_back(node);
      }
      };
    driver_pin->preOrder(find_steiner);
    // recover load pins' timing
    std::ranges::for_each(load_pins, [](Pin* pin) {
      pin->set_parent(nullptr);
      pin->set_children({});
      pin->set_sub_len(0);
      pin->set_slew_in(0);
      pin->set_cap_load(0);
      pin->set_required_snake(0);
      pin->set_net(nullptr);
      updatePinCap(pin);
      initLoadPinDelay(pin);
      });
    // release buffer and its pins
    auto* buffer = driver_pin->get_inst();
    delete buffer;
    // release steiner node
    std::ranges::for_each(to_be_removed, [](Node* node) { delete node; });
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
  void TimingPropagator::updatePinCap(Pin* pin)
  {
    double cap_load = 0;
    if (pin->isSinkPin()) {
      cap_load = CTSAPIInst.getSinkCap(pin->get_name());
    }
    if (pin->isBufferPin()) {
      auto cell_name = pin->get_cell_master();
      if (cell_name.empty()) {
        return;
      }
      auto* lib = CTSAPIInst.getCellLib(cell_name);
      cap_load = lib->get_init_cap();
    }
    pin->set_cap_load(cap_load);
  }
  /**
   * @brief update all timing information
   *       propagate net_len, cap, slew, cell_delay, wire_delay
   *
   * @param net
   */
  void TimingPropagator::update(Net* net)
  {
    if (net == nullptr) {
      return;
    }
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
    driver_pin->postOrder(updateNetLen<Node>);
  }
  /**
   * @brief propagate cap by post order, and update load pin's cap
   *
   * @param net
   */
  void TimingPropagator::capPropagate(Net* net)
  {
    std::ranges::for_each(net->get_load_pins(), [](Pin* pin) {
      double cap_load = 0;
      if (pin->isSinkPin()) {
        cap_load = CTSAPIInst.getSinkCap(pin->get_name());
      }
      if (pin->isBufferPin()) {
        auto cell_name = pin->get_cell_master();
        if (cell_name.empty()) {
          return;
        }
        auto* lib = CTSAPIInst.getCellLib(cell_name);
        cap_load = lib->get_init_cap();
      }
      pin->set_cap_load(cap_load);
      });
    auto* driver_pin = net->get_driver_pin();
    driver_pin->postOrder(updateCapLoad<Node>);
  }
  /**
   * @brief propagate slew by pre order
   *
   * @param net
   */
  void TimingPropagator::slewPropagate(Net* net)
  {
    auto* driver_pin = net->get_driver_pin();
    auto cell_name = driver_pin->get_cell_master();
    if (cell_name.empty()) {
      return;
    }
    auto* lib = CTSAPIInst.getCellLib(cell_name);
    auto slew_out = lib->calcSlew(driver_pin->get_cap_load());
    driver_pin->set_slew_in(slew_out);
    driver_pin->preOrder(updateSlewIn<Node>);
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
    driver_pin->postOrder(updateWireDelay<Node>);
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
    if (!driver_pin) {
      return;
    }
    auto* load_pin = inst->get_load_pin();
    auto slew_in = load_pin->get_slew_in();
    auto cap_load = driver_pin->get_cap_load();
    auto cell_name = inst->get_cell_master();
    if (cell_name.empty()) {
      return;
    }
    if (cap_load == 0) {
      inst->set_insert_delay(0);
      load_pin->set_min_delay(0);
      load_pin->set_max_delay(0);
      return;
    }
    auto* lib = CTSAPIInst.getCellLib(cell_name);
    auto insert_delay = lib->calcDelay(slew_in, cap_load);
    inst->set_insert_delay(insert_delay);
    auto min_delay = driver_pin->get_min_delay();
    auto max_delay = driver_pin->get_max_delay();
    load_pin->set_min_delay(insert_delay + min_delay);
    load_pin->set_max_delay(insert_delay + max_delay);
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
    auto delta = skew - skew_bound.value_or(_skew_bound);
    if (delta > 0 && delta < kEpsilon) {
      node->set_min_delay(node->get_max_delay() - skew_bound.value_or(_skew_bound));
      return true;
    }
    return skew <= skew_bound.value_or(_skew_bound);
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
    }
    else {
      auto* driver_pin = inst->get_driver_pin();
      if (driver_pin->get_children().empty()) {
        inst->set_insert_delay(0);
        pin->set_min_delay(0);
        pin->set_max_delay(0);
        return;
      }
      double insert_delay = 0;
      if (by_cell) {
        auto cell_name = pin->get_cell_master();
        if (cell_name.empty()) {
          return;
        }
        for (auto* lib : _delay_libs) {
          if (lib->get_cell_master() == cell_name) {
            auto cap_coef = lib->get_delay_coef().back();
            auto intercept = lib->getDelayIntercept();
            insert_delay = intercept + cap_coef * inst->getCapOut();
            break;
          }
        }
      }
      else {
        auto* max_lib = _delay_libs.back();
        auto cap_coef = max_lib->get_delay_coef().back();
        insert_delay = TimingPropagator::getMinInsertDelay() + cap_coef * inst->getCapOut();
      }
      inst->set_insert_delay(insert_delay);
      pin->set_min_delay(driver_pin->get_min_delay() + insert_delay);
      pin->set_max_delay(driver_pin->get_max_delay() + insert_delay);
    }
  }
  /**
   * @brief calculate elmore delay for LayerPattern::kSingle
   *       elmore_delay = r * l * (c * l / 2 + c_t)
   *
   * @param cap
   * @param len
   * @return double
   */
  double TimingPropagator::calcElmoreDelay(const double& cap, const double& len)
  {
    return _unit_res * len * (_unit_cap * len / 2 + cap);
  }
  /**
   * @brief calculate elmore delay for LayerPattern::kHV or LayerPattern::kVH
   *       elmore_delay_hv = r_h * x * (c_h * x / 2 + c_t) + r_v * y * (c_v * y / 2 + c_t + c_h * x)
   *       elmore_delay_vh = r_v * y * (c_v * y / 2 + c_t) + r_h * x * (c_h * x / 2 + c_t + c_v * y)
   *
   * @param cap
   * @param x
   * @param y
   * @return double
   */
  double TimingPropagator::calcElmoreDelay(const double& cap, const double& x, const double& y, const RCPattern& pattern)
  {
    double delay = 0;
    switch (pattern) {
    case RCPattern::kSingle:
      delay = calcElmoreDelay(cap, x + y);
      break;
    case RCPattern::kHV:
      delay = _unit_h_res * x * (_unit_h_cap * x / 2 + cap) + _unit_v_res * y * (_unit_v_cap * y / 2 + cap + _unit_h_cap * x);
      break;
    case RCPattern::kVH:
      delay = _unit_v_res * y * (_unit_v_cap * y / 2 + cap) + _unit_h_res * x * (_unit_h_cap * x / 2 + cap + _unit_v_cap * y);
      break;
    default:
      break;
    }
    return delay;
  }
  /**
   * @brief get unit cap by pattern
   *
   * @param pattern
   * @return double
   */
  double TimingPropagator::getUnitCap(const LayerPattern& pattern)
  {
    switch (pattern) {
    case LayerPattern::kH:
      return _unit_h_cap;
      break;
    case LayerPattern::kV:
      return _unit_v_cap;
      break;
    default:
      return _unit_cap;
      break;
    }
  }
  /**
   * @brief get unit res by pattern
   *
   * @param pattern
   * @return double
   */
  double TimingPropagator::getUnitRes(const LayerPattern& pattern)
  {
    switch (pattern) {
    case LayerPattern::kH:
      return _unit_h_res;
      break;
    case LayerPattern::kV:
      return _unit_v_res;
      break;
    default:
      return _unit_res;
      break;
    }
  }
}  // namespace icts