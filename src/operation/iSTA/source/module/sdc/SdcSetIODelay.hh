// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
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

  SdcSetIODelay* copy() {
    auto* copy_io_delay = new SdcSetIODelay(get_constrain_name(), get_clock_name(), get_delay_value());
    copy_io_delay->set_rise(isRise());
    copy_io_delay->set_fall(isFall());
    copy_io_delay->set_max(isMax());
    copy_io_delay->set_min(isMin());
    if(isClockFall()) {
      copy_io_delay->set_clock_fall();
    }

    std::set<DesignObject*> objs(_objs);
    copy_io_delay->set_objs(std::move(objs));
    return copy_io_delay;
  }
  void set_rise(bool is_set) { _rise = is_set; }
  void set_fall(bool is_set) { _fall = is_set; }
  void set_max(bool is_set) { _max = is_set; }
  void set_min(bool is_set) { _min = is_set; }
  void set_clock_fall() { _clock_fall = 1; }
  void set_add() { _add = 1; }
  unsigned isRise() const { return _rise; }
  unsigned isFall() const { return _fall; }
  unsigned isMax() const { return _max; }
  unsigned isMin() const { return _min; }
  unsigned isClockFall() const { return _clock_fall; }
  unsigned isAdd() const { return _add; }

  void set_objs(std::set<DesignObject*>&& objs) { _objs = std::move(objs); }
  auto& get_objs() { return _objs; }

  const char* get_clock_name() { return _clock_name.c_str(); }
  double get_delay_value() const { return _delay_value; }
  void set_delay_value(double delay_value) { _delay_value = delay_value; }

 private:
  unsigned _rise : 1;
  unsigned _fall : 1;
  unsigned _max : 1;
  unsigned _min : 1;
  unsigned _clock_fall : 1;
  unsigned _add : 1;
  unsigned _reserved : 26;

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
