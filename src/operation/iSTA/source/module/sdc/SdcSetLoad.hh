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
 * @file SdcSetLoad.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The file is the set_load class of sdc.
 * @version 0.1
 * @date 2021-04-14
 */
#pragma once

#include <set>
#include <utility>

#include "SdcCommand.hh"
#include "netlist/Netlist.hh"

namespace ista {

/**
 * @brief The set_load constrain.
 *
 */
class SdcSetLoad : public SdcIOConstrain {
 public:
  SdcSetLoad(const char* constrain_name, double load_value);
  ~SdcSetLoad() override = default;

  unsigned isSetLoad() override { return 1; }

  void set_rise() { _rise = 1; }
  void set_fall() { _fall = 1; }
  void set_max() { _max = 1; }
  void set_min() { _min = 1; }
  void set_pin_load() { _pin_load = 1; }
  void set_wire_load() { _wire_load = 1; }
  void set_subtract_pin_load() { _subtract_pin_load = 1; }
  void set_allow_negative_load() { _allow_negative_load = 1; }
  unsigned isRise() const { return _rise; }
  unsigned isFall() const { return _fall; }
  unsigned isMax() const { return _max; }
  unsigned isMin() const { return _min; }
  unsigned isPinLoad() const { return _pin_load; }
  unsigned isWireLoad() const { return _wire_load; }
  unsigned isSubtractPinLoad() const { return _subtract_pin_load; }
  unsigned isAllowNegativeLoad() const { return _allow_negative_load; }

  void set_objs(std::set<DesignObject*>&& objs) { _objs = std::move(objs); }
  auto& get_objs() { return _objs; }

  double get_load_value() { return _load_value; }

 private:
  unsigned _rise : 1;
  unsigned _fall : 1;
  unsigned _max : 1;
  unsigned _min : 1;
  unsigned _pin_load : 1;
  unsigned _wire_load : 1;
  unsigned _subtract_pin_load : 1;
  unsigned _allow_negative_load : 1;
  unsigned _reserved : 24;

  std::set<DesignObject*> _objs;  //!< The clock source object.

  double _load_value;
};

}  // namespace ista
