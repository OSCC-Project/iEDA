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
 * @file sdcClock.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The file is the clock class obj of sdc.
 * @version 0.1
 * @date 2020-11-27
 */

#pragma once

#include <set>
#include <utility>

#include "SdcCommand.hh"
#include "netlist/DesignObject.hh"
#include "netlist/Netlist.hh"

namespace ista {

/**
 * @brief The sdc clock.
 *
 */
class SdcClock : public SdcCommandObj {
 public:
  using SdcWaveform = Vector<double>;
  explicit SdcClock(const char* clock_name);
  ~SdcClock() override = default;

  unsigned isClock() override { return 1; }
  void set_clock_name(const char* clock_name) { _clock_name = clock_name; }
  const char* get_clock_name() { return _clock_name.c_str(); }

  void set_period(double period) { _period = period; }
  [[nodiscard]] double get_period() const { return _period; }
  void set_objs(std::set<DesignObject*>&& objs) { _objs = std::move(objs); }
  std::set<DesignObject*>& get_objs() { return _objs; }

  SdcWaveform& get_edges() { return _edges; }
  void set_edges(SdcWaveform&& edges) { _edges = std::move(edges); }

  void set_comment(std::string&& comment) { _tracking_comment = comment; }
  [[nodiscard]] std::string get_comment() const { return _tracking_comment; }

  [[nodiscard]] virtual bool isGenerateClock() const { return false; }

  [[nodiscard]] unsigned isPropagatedClock() const { return _is_propagated; }
  void set_is_propagated() { _is_propagated = 1; }

 private:
  std::string _clock_name;        //!< The clock name.
  double _period;                 //!< The clock period, unit is ns.
  std::set<DesignObject*> _objs;  //!< The clock source object.
  SdcWaveform _edges;  //!< The waveform edges, one rise edge, one fall edge.
  unsigned _is_propagated : 1 = 0;
  unsigned _reserved : 31 = 0;

  std::string _tracking_comment;  //!< The comment for debug.
};

/**
 * @brief The sdc generate clock.
 *
 */
class SdcGenerateCLock : public SdcClock {
 public:
  explicit SdcGenerateCLock(const char* clock_name);
  ~SdcGenerateCLock() override = default;

  [[nodiscard]] bool isGenerateClock() const override { return true; }

  bool isSameSource(const char* name) const {
    return Str::equal(name, _source_name.c_str());
  }
  void set_source_name(const char* source_name) { _source_name = source_name; }
  const char* get_source_name() const { return _source_name.c_str(); }

  void set_is_need_update_source_clock() { _is_need_update_source_clock = 1; }
  bool isNeedUpdateSourceClock() const { return _is_need_update_source_clock; }

  void set_is_waveform_inv() { _is_waveform_inv = 1; }
  bool isWaveformInv() const { return _is_waveform_inv; }

  void set_source_pins(std::set<DesignObject*> source_pins) {
    _source_pins = std::move(source_pins);
  }
  std::set<DesignObject*> get_source_pins() { return _source_pins; }

  void set_divide_by(int divide_by) { _divide_by = divide_by; }
  int get_divide_by() const { return _divide_by; }

 private:
  std::string _source_name;

  // need update source clock later after clock propagation.
  unsigned _is_need_update_source_clock : 1 = 0;
  unsigned _is_waveform_inv : 1 = 0;
  unsigned _reserved : 30 = 0;

  std::set<DesignObject*> _source_pins;
  int _divide_by = 1;
};

/**
 * @brief The all clocks sdc obj.
 *
 */
class SdcAllClocks : public SdcCommandObj {
 public:
  SdcAllClocks() = default;
  ~SdcAllClocks() override = default;

  void addClock(SdcClock* clock) { _clocks.push_back(clock); }
  auto& get_clocks() { return _clocks; }

  unsigned isAllClock() override { return 1; }

 private:
  std::vector<SdcClock*> _clocks;  //!< The clocks.
};

/**
 * @brief The clock group obj.
 *
 */
class SdcClockGroup {
 public:
  SdcClockGroup() = default;
  ~SdcClockGroup() = default;
  SdcClockGroup(SdcClockGroup&& other) noexcept = default;
  SdcClockGroup& operator=(SdcClockGroup&& rhs) noexcept = default;

  void addClock(std::string&& clock) {
    _clock_group.emplace_back(std::move(clock));
  }
  auto& get_clock_group() { return _clock_group; }
  bool isClockInGroup(std::string& clock_name) {
    return _clock_group.end() != std::find_if(_clock_group.begin(),
                                              _clock_group.end(),
                                              [&clock_name](auto& the_clock) {
                                                return clock_name == the_clock;
                                              });
  }

 private:
  std::vector<std::string> _clock_group;  //!< The clock group.
};

/**
 * @brief The set_clock_groups obj.
 *
 */
class SdcClockGroups : public SdcCommandObj {
 public:
  explicit SdcClockGroups(std::string&& group_name)
      : _group_name(std::move(group_name)) {}
  ~SdcClockGroups() override = default;

  void addClockGroup(SdcClockGroup&& clock_group) {
    _clock_groups.emplace_back(std::move(clock_group));
  }

  bool isInAsyncGroup(std::string& clock_name1, std::string& clock_name2) {
    auto it1 = std::find_if(_clock_groups.begin(), _clock_groups.end(),
                            [&clock_name1](auto& clock_group) {
                              return clock_group.isClockInGroup(clock_name1);
                            });

    if (it1 != _clock_groups.end()) {
      auto it2 = std::find_if(_clock_groups.begin(), _clock_groups.end(),
                              [&clock_name2](auto& clock_group) {
                                return clock_group.isClockInGroup(clock_name2);
                              });
      return it1 != it2;
    }

    return false;
  }

 private:
  std::vector<SdcClockGroup> _clock_groups;
  std::string _group_name;
};

}  // namespace ista
