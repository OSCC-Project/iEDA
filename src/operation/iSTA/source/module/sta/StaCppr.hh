/**
 * @file StaCppr.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The cppr is clock path pessimism removal.The class of cppr find the
 * common launch/latch clock path point.
 * @version 0.1
 * @date 2021-03-24
 */

#pragma once

#include "StaFunc.hh"

namespace ista {

/**
 * @brief The clock path pessimism removal class.
 *
 */
class StaCppr : public StaFunc {
 public:
  StaCppr(StaClockData* launch_data, StaClockData* capture_data);
  ~StaCppr() override = default;

  virtual unsigned operator()(StaClock* the_clock);
  int get_cppr() const { return _cppr; }

 private:
  StaVertex* getLCA(StaVertex* root, StaVertex* clock_end1,
                    StaVertex* clock_end2);
  StaClockData* _launch_data;
  StaClockData* _capture_data;
  StaVertex* _common_point = nullptr;
  int _cppr = 0;
};
}  // namespace ista
