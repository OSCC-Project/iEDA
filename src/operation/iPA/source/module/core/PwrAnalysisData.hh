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
 * @file PwrAnalysisData.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief The class of power analysis data, including leakage power, internal
 * power and switch power.
 * @version 0.1
 * @date 2023-04-12
 */
#pragma once
#include "netlist/DesignObject.hh"

namespace ipower {
using ista::DesignObject;
/**
 * @brief The base class of power analysis data.
 *
 */
class PwrAnalysisData {
 public:
  explicit PwrAnalysisData(DesignObject* design_obj)
      : _design_obj(design_obj) {}
  virtual ~PwrAnalysisData() = default;

  [[nodiscard]] virtual unsigned isLeakageData() const { return 0; }
  [[nodiscard]] virtual unsigned isInternalData() const { return 0; }
  [[nodiscard]] virtual unsigned isSwitchData() const { return 0; }

  [[nodiscard]] virtual double getPowerDataValue() const { return 0; }

  auto* get_design_obj() { return _design_obj; }

  void set_nom_voltage(double nom_voltage) { _nom_voltage = nom_voltage; }
  [[nodiscard]] double get_nom_voltage() const { return _nom_voltage; }

 protected:
  DesignObject* _design_obj;
  double _nom_voltage = 0.0;  //!< The power analysis nominal voltage.
};

/**
 * @brief The class of leakage power data.
 *
 */
class PwrLeakageData : public PwrAnalysisData {
 public:
  PwrLeakageData(DesignObject* design_obj, double leakage_power)
      : PwrAnalysisData(design_obj), _leakage_power(leakage_power) {}
  ~PwrLeakageData() override = default;

  [[nodiscard]] unsigned isLeakageData() const override { return 1; }
  [[nodiscard]] double get_leakage_power() const { return _leakage_power; }

  [[nodiscard]] double getPowerDataValue() const override {
    return _leakage_power;
  }

 private:
  double _leakage_power; //!< unit is W.
};

/**
 * @brief The class of internal power data.
 *
 */
class PwrInternalData : public PwrAnalysisData {
 public:
  PwrInternalData(DesignObject* design_obj, double internal_power)
      : PwrAnalysisData(design_obj), _internal_power(internal_power) {}
  ~PwrInternalData() override = default;

  [[nodiscard]] unsigned isInternalData() const override { return 1; }
  [[nodiscard]] double get_internal_power() const { return _internal_power; }

  [[nodiscard]] double getPowerDataValue() const override {
    return _internal_power;
  }

 private:
  double _internal_power; //!< unit is W.
};

/**
 * @brief The class of switch power data.
 *
 */
class PwrSwitchData : public PwrAnalysisData {
 public:
  PwrSwitchData(DesignObject* design_obj, double switch_power)
      : PwrAnalysisData(design_obj), _switch_power(switch_power) {}
  ~PwrSwitchData() override = default;

  [[nodiscard]] unsigned isSwitchData() const override { return 1; }
  [[nodiscard]] double get_switch_power() const { return _switch_power; }

  [[nodiscard]] double getPowerDataValue() const override {
    return _switch_power;
  }

 private:
  double _switch_power; //!< Unit is W.
};
}  // namespace ipower