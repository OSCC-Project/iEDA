/**
 * @file StaClockPropagation.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class of clock propagation.
 * @version 0.1
 * @date 2021-02-19
 *
 * @copyright Copyright (c) 2021
 *
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
  enum class PropType { kIdealClockProp, kNormalClockProp };

  explicit StaClockPropagation(PropType prop_type) : _prop_type(prop_type) {}
  ~StaClockPropagation() override = default;

  unsigned operator()(StaGraph* the_graph) override;

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
