// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file PwrGroupData.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief The class of power path group data, used for power report.
 * @version 0.1
 * @date 2023-04-25
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
  auto get_obj() { return _obj; }

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
  [[nodiscard]] double get_total_power() const {
    return _internal_power + _switch_power + _leakage_power;
  }

  void set_nom_voltage(double nom_voltage) { _nom_voltage = nom_voltage; }
  [[nodiscard]] double get_nom_voltage() const { return _nom_voltage; }

 private:
  PwrGroupType _group_type;  //!< The group type.

  double _internal_power = 0.0; //!< unit is W.
  double _switch_power = 0.0;  //!< unit is W.
  double _leakage_power = 0.0;  //!< unit is W.

  double _nom_voltage = 0.0; //!< unit is V.

  DesignObject* _obj;
};

};  // namespace ipower
