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
 * @file TimingPropagator.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
#include <concepts>

#include "CtsCellLib.hh"
#include "CtsConfig.hh"
#include "Inst.hh"
#include "Net.hh"
#include "Node.hh"
#include "Pin.hh"

namespace icts {
/**
 * @brief Concept of timing propagator
 *
 */
template <typename T>
concept IntAble = std::integral<T>;

template <typename T>
concept FloatAble = std::floating_point<T>;

template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

template <typename T>
concept LocAble = requires(T node)
{
  {
    node.get_location().x()
  }
  ->Numeric;
  {
    node.get_location().y()
  }
  ->Numeric;
};

template <typename T>
concept ParentAble = LocAble<T>&& requires(T node)
{
  {
    node.get_parent()
  }
  ->std::same_as<T*>;
};

template <typename T>
concept ChildrenAble = LocAble<T>&& requires(T node)
{
  {
    node.get_children()
  }
  ->std::common_reference_with<const std::vector<T*>&>;
};

template <typename T>
concept TreeAble = LocAble<T>&& ParentAble<T>&& ChildrenAble<T>;

template <typename T>
concept NetLenAble = TreeAble<T>&& requires(T node, const double& sub_len)
{
  {
    node.get_sub_len()
  }
  ->std::convertible_to<double>;
  {
    node.set_sub_len(sub_len)
  }
  ->std::same_as<void>;
};

template <typename T>
concept CapAble = TreeAble<T>&& requires(T node, const double& cap_load)
{
  {
    node.get_cap_load()
  }
  ->std::convertible_to<double>;
  {
    node.set_cap_load(cap_load)
  }
  ->std::same_as<void>;
};

template <typename T>
concept SlewAble = CapAble<T>&& requires(T node, const double& slew_in)
{
  {
    node.get_slew_in()
  }
  ->std::convertible_to<double>;
  {
    node.set_slew_in(slew_in)
  }
  ->std::same_as<void>;
};

template <typename T>
concept DelayAble = CapAble<T>&& requires(T node, const double& delay)
{
  {
    node.get_min_delay()
  }
  ->std::convertible_to<double>;
  {
    node.set_min_delay(delay)
  }
  ->std::same_as<void>;
  {
    node.get_max_delay()
  }
  ->std::convertible_to<double>;
  {
    node.set_max_delay(delay)
  }
  ->std::same_as<void>;
};

template <typename T>
concept SnakeAble = requires(T node, const double& required_snake)
{
  {
    node.get_required_snake()
  }
  ->std::convertible_to<double>;
  {
    node.set_required_snake(required_snake)
  }
  ->std::same_as<void>;
};

template <typename T>
concept LoadPinAble = requires(T node)
{
  {
    node.isPin()
  }
  ->std::same_as<bool>;
  {
    node.isLoad()
  }
  ->std::same_as<bool>;
};

/**
 * @brief Timing propagator
 *       Propagate timing information from root to leaf
 *       Support: - cap, slew, delay calculation, propagation, update
 *
 */
class TimingPropagator
{
 public:
  TimingPropagator() = delete;
  ~TimingPropagator() = default;

  static void init();
  // net based
  static Net* genNet(const std::string& net_name, Pin* driver_pin, const std::vector<Pin*>& load_pins = {});
  static void resetNet(Net* net);
  static void updateLoads(Net* net);
  static void updatePinCap(Pin* pin);
  static void update(Net* net);
  static void netLenPropagate(Net* net);
  static void capPropagate(Net* net);
  static void slewPropagate(Net* net);
  static void cellDelayPropagate(Net* net);
  static void wireDelayPropagate(Net* net);
  // inst based
  static void updateCellDelay(Inst* inst);
  static double calcSkew(Node* node);
  static bool skewFeasible(Node* node, const std::optional<double>& skew_bound = std::nullopt);
  // pin based
  static void initLoadPinDelay(Pin* pin, const bool& by_cell = false);
  // timing calc
  static double calcElmoreDelay(const double& cap, const double& len);
  static double calcElmoreDelay(const double& cap, const double& x, const double& y, const RCPattern& pattern = RCPattern::kSingle);
  // info getter
  static double getUnitCap(const LayerPattern& pattern = LayerPattern::kNone);
  static double getUnitRes(const LayerPattern& pattern = LayerPattern::kNone);
  // get
  static double getSkewBound() { return _skew_bound; }
  static int getDbUnit() { return _db_unit; }
  static double getMaxBufTran() { return _max_buf_tran; }
  static double getMaxSinkTran() { return _max_sink_tran; }
  static double getMaxCap() { return _max_cap; }
  static int getMaxFanout() { return _max_fanout; }
  static double getMinLength() { return _min_length; }
  static double getMaxLength() { return _max_length; }
  static double getMinInsertDelay() { return _min_insert_delay; }
  static icts::CtsCellLib* getMinSizeLib() { return _delay_libs.front(); }
  static icts::CtsCellLib* getMaxSizeLib() { return _delay_libs.back(); }
  static icts::CtsCellLib* getRootSizeLib() { return _root_lib; }
  static std::string getMinSizeCell() { return getMinSizeLib()->get_cell_master(); }
  static std::string getMaxSizeCell() { return getMaxSizeLib()->get_cell_master(); }
  static std::string getRootSizeCell() { return getRootSizeLib()->get_cell_master(); }
  static std::vector<icts::CtsCellLib*> getDelayLibs() { return _delay_libs; }
  // node based
  /**
   * @brief update net's wirelength
   *
   * @tparam T
   * @param node
   */
  template <NetLenAble T>
  static void updateNetLen(T* node)
  {
    auto net_len = calcNetLen(node);
    node->set_sub_len(net_len);
  }
  /**
   * @brief update pin's cap load and cap out
   *
   * @tparam T
   * @param node
   * @param pattern
   */
  template <CapAble T>
  static void updateCapLoad(T* node, const RCPattern& pattern = RCPattern::kSingle)
  {
    auto cap_load = calcCapLoad(node, pattern);
    node->set_cap_load(cap_load);
  }
  /**
   * @brief update slew, for slew constraint and insertion delay calculation
   *       driver pin: slew_out = a * [cap_load] + b, the function is from liberty fitting
   *       load pin or steiner node: slew_out = [slew_in]
   *                                 slew_in = sqrt([parent's slew_out]^2 + [slew_wire]^2)
   *
   * @tparam T
   * @param node
   * @param pattern
   */
  template <SlewAble T>
  static void updateSlewIn(T* node, const RCPattern& pattern = RCPattern::kSingle)
  {
    auto calc_slew_in = [&node](T* child) {
      auto slew_ideal = calcIdealSlew(node, child);
      double slew_in = std::sqrt(std::pow(node->get_slew_in(), 2) + std::pow(slew_ideal, 2));
      child->set_slew_in(slew_in);
    };
    std::ranges::for_each(node->get_children(), calc_slew_in);
  }
  /**
   * @brief update wire delay between node and its child
   *       wire delay: elmore model delay
   *
   * @tparam T
   * @param node
   * @param pattern
   */
  template <DelayAble T>
  static void updateWireDelay(T* node, const RCPattern& pattern = RCPattern::kSingle)
  {
    if (node->get_children().empty()) {
      return;
    }
    double min_delay = std::numeric_limits<double>::max();
    double max_delay = std::numeric_limits<double>::min();
    auto calc_delay = [&min_delay, &max_delay, &node, &pattern](T* child) {
      auto delay = calcElmoreDelay(node, child, pattern);
      min_delay = std::min(min_delay, delay + child->get_min_delay());
      max_delay = std::max(max_delay, delay + child->get_max_delay());
    };
    std::ranges::for_each(node->get_children(), calc_delay);

    if constexpr (LoadPinAble<T>) {
      if (node->isLoad()) {
        min_delay = std::min(min_delay, node->get_min_delay());
        max_delay = std::max(max_delay, node->get_max_delay());
      }
    }
    node->set_min_delay(min_delay);
    node->set_max_delay(max_delay);
  }
  /**
   * @brief calculate downstream wirelength
   *
   * @tparam T
   * @param node
   * @return double
   */
  template <NetLenAble T>
  static double calcNetLen(T* node)
  {
    double net_len = 0;
    auto accumulate_net_len = [&net_len, &node](T* child) {
      net_len += calcLen(node, child) + child->get_sub_len();
      if constexpr (SnakeAble<T>) {
        net_len += child->get_required_snake();
      }
    };
    std::ranges::for_each(node->get_children(), accumulate_net_len);
    return net_len;
  }
  /**
   * @brief calculate capacitance
   *
   * @tparam T
   * @param node
   * @param pattern
   * @return double
   */
  template <CapAble T>
  static double calcCapLoad(T* node, const RCPattern& pattern = RCPattern::kSingle)
  {
    double cap_load = 0;
    if constexpr (LoadPinAble<T>) {
      if (node->isLoad()) {
        cap_load = node->get_cap_load();
      }
    }
    auto accumulate_cap = [&cap_load, &node, &pattern](T* child) {
      switch (pattern) {
        // normal rc pattern
        case RCPattern::kSingle:
          cap_load += _unit_cap * calcLen(node, child, LayerPattern::kNone) + child->get_cap_load();
          if constexpr (SnakeAble<T>) {
            cap_load += _unit_h_cap * child->get_required_snake();
          }
          break;
        // HV or VH pattern
        default:
          cap_load += _unit_h_cap * calcLen(node, child, LayerPattern::kH) + _unit_v_cap * calcLen(node, child, LayerPattern::kV)
                      + child->get_cap_load();
          if constexpr (SnakeAble<T>) {
            cap_load += _unit_h_cap * child->get_required_snake();
          }
          break;
      }
    };
    std::ranges::for_each(node->get_children(), accumulate_cap);
    return cap_load;
  }
  /**
   * @brief calculate ideal slew (wire slew)
   *       ideal_slew = log(9) * elmore_delay
   *
   * @tparam T
   * @param parent
   * @param child
   * @param pattern
   * @return double
   */
  template <CapAble T>
  static double calcIdealSlew(T* parent, T* child, const RCPattern& pattern = RCPattern::kSingle)
  {
    return std::log(9) * calcElmoreDelay(parent, child, pattern);
  }
  /**
   * @brief calculate elmore delay
   *
   * @tparam T
   * @param parent
   * @param child
   * @param pattern
   * @return double
   */
  template <CapAble T>
  static double calcElmoreDelay(T* parent, T* child, const RCPattern& pattern = RCPattern::kSingle)
  {
    double delay = 0;
    auto cap_load = child->get_cap_load();
    switch (pattern) {
      // normal rc pattern
      case RCPattern::kSingle: {
        auto len = calcLen(parent, child);
        if constexpr (SnakeAble<T>) {
          delay = calcElmoreDelay(cap_load, len + child->get_required_snake());
        } else {
          delay = calcElmoreDelay(cap_load, len);
        }
        break;
      }
      // HV or VH pattern
      default: {
        auto x = calcLen(parent, child, LayerPattern::kH);
        auto y = calcLen(parent, child, LayerPattern::kV);
        delay = calcElmoreDelay(cap_load, x, y, pattern);
        if constexpr (SnakeAble<T>) {
          delay += calcElmoreDelay(cap_load + _unit_h_cap * x + _unit_v_cap * y, child->get_required_snake(), 0, pattern);
        }
        break;
      }
    }
    return delay;
  }
  /**
   * @brief calculate wirelength between parent and child (not consider snake)
   *
   * @tparam T
   * @param parent
   * @param child
   * @param pattern
   * @return auto
   */
  template <LocAble T>
  static auto calcLen(T& parent, T& child, const LayerPattern& pattern = LayerPattern::kNone)
  {
    auto parent_loc = parent.get_location();
    auto child_loc = child.get_location();
    auto len = calcLen(parent_loc, child_loc, pattern);
    return len;
  }
  template <LocAble T>
  static auto calcLen(T* parent, T* child, const LayerPattern& pattern = LayerPattern::kNone)
  {
    return calcLen(*parent, *child, pattern);
  }
  /**
   * @brief calculate manhattan wire length between p1 and p2
   *       if T is integer,  l = D(n1,n2) / db_unit, for design core/die coordinate
   *       if T is float,    l = D(n1,n2)          , for absolute coordinate
   *
   * @tparam T
   * @param p1
   * @param p2
   * @param pattern
   * @return auto
   */
  template <IntAble T>
  static auto calcLen(const CtsPoint<T>& p1, const CtsPoint<T>& p2, const LayerPattern& pattern = LayerPattern::kNone)
  {
    return 1.0 * calcDist(p1, p2, pattern) / _db_unit;
  }
  template <FloatAble T>
  static auto calcLen(const CtsPoint<T>& p1, const CtsPoint<T>& p2, const LayerPattern& pattern = LayerPattern::kNone)
  {
    return 1.0 * calcDist(p1, p2, pattern);
  }
  /**
   * @brief calculate manhattan dist between parent and child (not consider snake)
   *
   * @tparam T
   * @param parent
   * @param child
   * @param pattern
   * @return auto
   */
  template <LocAble T>
  static auto calcDist(T& parent, T& child, const LayerPattern& pattern = LayerPattern::kNone)
  {
    auto parent_loc = parent.get_location();
    auto child_loc = child.get_location();
    auto dist = calcDist(parent_loc, child_loc, pattern);
    return dist;
  }
  template <LocAble T>
  static auto calcDist(T* parent, T* child, const LayerPattern& pattern = LayerPattern::kNone)
  {
    return calcDist(*parent, *child, pattern);
  }
  /**
   * @brief calculate manhattan dist between p1 and p2
   *       D(n1,n2) = |x1 - x2| + |y1 - y2|
   *
   * @param p1
   * @param p2
   * @return T
   */
  template <Numeric T>
  static T calcDist(const CtsPoint<T>& p1, const CtsPoint<T>& p2, const LayerPattern& pattern = LayerPattern::kNone)
  {
    T dist = 0;
    switch (pattern) {
      case LayerPattern::kNone:
        dist = std::abs(p1.x() - p2.x()) + std::abs(p1.y() - p2.y());
        break;
      case LayerPattern::kH:
        dist = std::abs(p1.x() - p2.x());
        break;
      case LayerPattern::kV:
        dist = std::abs(p1.y() - p2.y());
        break;
      default:
        break;
    }
    return dist;
  }

  constexpr static double kEpsilon = 1e-6;

 private:
  static double _unit_cap;  // pf
  static double _unit_res;  // kilo-ohm
  static double _unit_h_cap;
  static double _unit_h_res;
  static double _unit_v_cap;
  static double _unit_v_res;
  static double _skew_bound;
  static int _db_unit;
  static double _max_buf_tran;
  static double _max_sink_tran;
  static double _max_cap;
  static int _max_fanout;
  static double _min_length;
  static double _max_length;
  static double _min_insert_delay;
  static std::vector<icts::CtsCellLib*> _delay_libs;
  static icts::CtsCellLib* _root_lib;
};

}  // namespace icts