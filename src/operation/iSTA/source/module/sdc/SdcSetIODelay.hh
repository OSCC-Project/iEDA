/**
 * @file SdcSetIODelay.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The file is set io delay constrain class.
 * @version 0.1
 * @date 2021-05-24
 */
#pragma once

#include <set>
#include <utility>

#include "SdcCommand.hh"
#include "netlist/Netlist.hh"

namespace ista {

/**
 * @brief The io delay of io constrain.
 *
 */
class SdcSetIODelay : public SdcIOConstrain {
 public:
  SdcSetIODelay(const char* constrain_name, const char* clock_name,
                double delay_value);
  ~SdcSetIODelay() override = default;

  void set_rise(bool is_set) { _rise = is_set; }
  void set_fall(bool is_set) { _fall = is_set; }
  void set_max(bool is_set) { _max = is_set; }
  void set_min(bool is_set) { _min = is_set; }
  void set_clock_fall() { _clock_fall = 1; }
  unsigned isRise() const { return _rise; }
  unsigned isFall() const { return _fall; }
  unsigned isMax() const { return _max; }
  unsigned isMin() const { return _min; }
  unsigned isClockFall() const { return _clock_fall; }

  void set_objs(std::set<DesignObject*>&& objs) { _objs = std::move(objs); }
  auto& get_objs() { return _objs; }

  const char* get_clock_name() { return _clock_name.c_str(); }
  double get_delay_value() const { return _delay_value; }

 private:
  unsigned _rise : 1;
  unsigned _fall : 1;
  unsigned _max : 1;
  unsigned _min : 1;
  unsigned _clock_fall : 1;
  unsigned _reserved : 27;

  std::string _clock_name;
  double _delay_value;

  std::set<DesignObject*> _objs;  //!< The clock source object.
};

/**
 * @brief The input delay of io constrain.
 *
 */
class SdcSetInputDelay : public SdcSetIODelay {
 public:
  SdcSetInputDelay(const char* constrain_name, const char* clock_name,
                   double delay_value);
  ~SdcSetInputDelay() override = default;

  unsigned isSetInputDelay() override { return 1; }
};

/**
 * @brief The output delay of io constrain.
 *
 */
class SdcSetOutputDelay : public SdcSetIODelay {
 public:
  SdcSetOutputDelay(const char* constrain_name, const char* clock_name,
                    double delay_value);
  ~SdcSetOutputDelay() override = default;

  unsigned isSetOutputDelay() override { return 1; }
};

}  // namespace ista
