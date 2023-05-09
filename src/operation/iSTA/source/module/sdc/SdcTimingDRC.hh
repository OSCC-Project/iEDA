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
 * @file SdcTimingDRC.hh
 * @author Wang Hao (harry0789@qq.com)
 * @brief
 * @version 0.1
 * @date 2021-10-06
 */
#pragma once

#include <set>
#include <utility>

#include "SdcCollection.hh"
#include "SdcCommand.hh"
#include "netlist/Netlist.hh"

namespace ista {

/**
 * @brief TimingDRC
 *
 * I don't really understand what is this
 *
 */
class SdcTimingDRC : public SdcCommandObj {
 public:
  explicit SdcTimingDRC(double drc_val) : _drc_val(drc_val) {}

  ~SdcTimingDRC() override = default;

  unsigned isTimingDRC() override { return 1; }

  virtual unsigned isMaxTransition() { return 0; }
  virtual unsigned isMaxCap() { return 0; }
  virtual unsigned isMaxFanout() { return 0; }

  virtual unsigned isRise() const {
    LOG_FATAL << "not supported";
    return 0;
  }
  virtual unsigned isFall() const {
    LOG_FATAL << "not supported";
    return 0;
  }

  virtual unsigned isClockPath() const {
    LOG_FATAL << "not supported";
    return 0;
  }

  virtual unsigned isDataPath() const {
    LOG_FATAL << "not supported";
    return 0;
  }

  void set_objs(std::set<SdcCollectionObj>&& objs) { _objs = std::move(objs); }
  auto& get_objs() { return _objs; }

  [[nodiscard]] double get_drc_val() const { return _drc_val; }

 private:
  double _drc_val;
  std::set<SdcCollectionObj> _objs;  //!< The source object.
};

/**
 * @brief The SetMaxTransition
 *
 */
class SetMaxTransition : public SdcTimingDRC {
 public:
  explicit SetMaxTransition(double transition_value);

  ~SetMaxTransition() override = default;

  unsigned isMaxTransition() override { return 1; }

  void set_is_rise() { _is_rise = 1; }
  void set_is_fall() { _is_fall = 1; }

  [[nodiscard]] unsigned isRise() const override { return _is_rise; }
  [[nodiscard]] unsigned isFall() const override { return _is_fall; }

  void set_is_clock_path() { _is_clock_path = 1; }
  void set_is_data_path() { _is_data_path = 1; }

  [[nodiscard]] unsigned isClockPath() const override { return _is_clock_path; }
  [[nodiscard]] unsigned isDataPath() const override { return _is_data_path; }

 private:
  unsigned _is_clock_path : 1;
  unsigned _is_data_path : 1;
  unsigned _is_rise : 1;
  unsigned _is_fall : 1;
  unsigned _reserved : 28;
};

/**
 * @brief The SetMaxCapacitance
 *
 */
class SetMaxCapacitance : public SdcTimingDRC {
 public:
  explicit SetMaxCapacitance(double capacitance_value);

  ~SetMaxCapacitance() override = default;

  unsigned isMaxCap() override { return 1; }

  void set_is_rise() { _is_rise = 1; }
  void set_is_fall() { _is_fall = 1; }

  [[nodiscard]] unsigned isRise() const override { return _is_rise; }
  [[nodiscard]] unsigned isFall() const override { return _is_fall; }

  void set_is_clock_path() { _is_clock_path = 1; }
  void set_is_data_path() { _is_data_path = 1; }

  [[nodiscard]] unsigned isClockPath() const override { return _is_clock_path; }
  [[nodiscard]] unsigned isDataPath() const override { return _is_data_path; }

 private:
  unsigned _is_clock_path : 1;
  unsigned _is_data_path : 1;
  unsigned _is_rise : 1;
  unsigned _is_fall : 1;
  unsigned _reserved : 28;
};

/**
 * @brief The io delay of io constrain.
 *
 */
class SetMaxFanout : public SdcTimingDRC {
 public:
  explicit SetMaxFanout(double fanout_value);

  ~SetMaxFanout() override = default;

  unsigned isMaxFanout() override { return 1; }
};

}  // namespace ista
