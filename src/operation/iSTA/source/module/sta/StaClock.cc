/**
 * @file StaClock.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of sta clock.
 * @version 0.1
 * @date 2021-02-17
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "StaClock.hh"

#include <utility>

#include "StaFunc.hh"

namespace ista {

StaClock::StaClock(const char* clock_name, ClockType clock_type, int period)
    : _clock_name(Str::copy(clock_name)),
      _clock_type(clock_type),
      _period(period) {}
StaClock::~StaClock() {
  Str::free(_clock_name);
  _clock_name = nullptr;
}
StaClock::StaClock(StaClock&& other)
    : _clock_name(other._clock_name),
      _clock_vertexes(std::move(other._clock_vertexes)),
      _clock_type(other._clock_type),
      _period(other._period),
      _wave_form(std::move(other._wave_form)) {
  other._clock_name = nullptr;
}
StaClock& StaClock::operator=(StaClock&& rhs) {
  if (this != &rhs) {
    _clock_name = rhs._clock_name;
    _clock_vertexes = std::move(rhs._clock_vertexes);
    _clock_type = rhs._clock_type;
    _period = rhs._period;
    _wave_form = std::move(rhs._wave_form);

    rhs._clock_name = nullptr;
  }

  return *this;
}

unsigned StaClock::exec(StaFunc& func) { return func(this); }

}  // namespace ista