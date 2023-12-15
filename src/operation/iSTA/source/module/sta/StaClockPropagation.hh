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
 * @file StaClockPropagation.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class of clock propagation.
 * @version 0.1
 * @date 2021-02-19
 */
#pragma once

#include "StaFunc.hh"

namespace ista {

/**
 * @brief The functor of clock propagation.
 *
 */
class StaClockPropagation : public StaFunc {
 public:
  enum class PropType {
    kIdealClockProp,
    kNormalClockProp,
    kUpdateGeneratedClockProp
  };

  explicit StaClockPropagation(PropType prop_type) : _prop_type(prop_type) {}
  ~StaClockPropagation() override = default;

  unsigned operator()(StaGraph* the_graph) override;
  // unsigned operator()(StaClock* the_clock) override;

  void updateSdcGeneratedClock();
  void set_propagate_clock(StaClock* propagate_clock) {
    _propagate_clock = propagate_clock;
  }

  [[nodiscard]] bool isIdealClock() const {
    return _propagate_clock->isIdealClockNetwork();
  }

 private:
  unsigned propagateClock(StaVertex* the_vertex, StaClockData* data1,
                          StaClockData* data2);
  unsigned propagateClock(StaArc* the_arc, StaClockData* data1,
                          StaClockData* data2);

  StaClock* _propagate_clock;
  PropType _prop_type;
};

}  // namespace ista
