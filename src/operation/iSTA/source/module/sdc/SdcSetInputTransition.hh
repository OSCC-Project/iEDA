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
 * @file SdcSetInputTransition.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The file is the set_input_transition class of sdc.
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
 * @brief The set_input_transition constrain.
 *
 */
class SdcSetInputTransition : public SdcIOConstrain {
 public:
  SdcSetInputTransition(const char* constrain_name, double transition_value);
  ~SdcSetInputTransition() override = default;

  unsigned isSetInputTransition() override { return 1; }

  void set_rise(bool is_set) { _rise = is_set; }
  void set_fall(bool is_set) { _fall = is_set; }
  void set_max(bool is_set) { _max = is_set; }
  void set_min(bool is_set) { _min = is_set; }
  [[nodiscard]] unsigned isRise() const { return _rise; }
  [[nodiscard]] unsigned isFall() const { return _fall; }
  [[nodiscard]] unsigned isMax() const { return _max; }
  [[nodiscard]] unsigned isMin() const { return _min; }

  void set_objs(std::set<DesignObject*>&& objs) { _objs = std::move(objs); }
  auto& get_objs() { return _objs; }

  [[nodiscard]] double get_transition_value() const {
    return _transition_value;
  }

 private:
  unsigned _rise : 1;
  unsigned _fall : 1;
  unsigned _max : 1;
  unsigned _min : 1;
  unsigned _reserved : 28;

  std::set<DesignObject*> _objs;  //!< The clock source object.

  double _transition_value;
};

}  // namespace ista
