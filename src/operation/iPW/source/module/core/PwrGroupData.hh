/**
 * @file PwrGroupData.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief The class of power path group data, used for power report.
 * @version 0.1
 * @date 2023-04-25
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include "include/PwrType.hh"
#include "netlist/DesignObject.hh"

namespace ipower {

/**
 * @brief report data by group.
 *
 */
class PwrGroupData {
 public:
  enum class PwrGroupType {
    kIOPad = 0,
    kMemory,
    kBlackBox,
    kClockNetwork,
    kRegister,
    kComb,
    kSeq,
    kUserDefine
  };

  PwrGroupData(PwrGroupType group_type, DesignObject* obj)
      : _group_type(group_type), _obj(obj) {}
  ~PwrGroupData() = default;

  auto get_group_type() { return _group_type; }

  void set_internal_power(double internal_power) {
    _internal_power = internal_power;
  }
  [[nodiscard]] double get_internal_power() const { return _internal_power; }

  void set_switch_power(double switch_power) { _switch_power = switch_power; }
  [[nodiscard]] double get_switch_power() const { return _switch_power; }

  void set_leakage_power(double leakage_power) {
    _leakage_power = leakage_power;
  }
  [[nodiscard]] double get_leakage_power() const { return _leakage_power; }

 private:
  PwrGroupType _group_type;  //!< The group type.

  double _internal_power = 0.0;
  double _switch_power = 0.0;
  double _leakage_power = 0.0;

  DesignObject* _obj;
};

};  // namespace ipower
