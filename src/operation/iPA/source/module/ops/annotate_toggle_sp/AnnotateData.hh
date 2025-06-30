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
 * @file AnnotateData.hh
 * @author shaozheqing (707005020@qq.com)
 * @brief The annotate data, which may from vcd,saif.
 * @version 0.1
 * @date 2023-01-11
 */
#pragma once

#include <gmpxx.h>

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "include/PwrType.hh"
#include "log/Log.hh"

namespace ipower {

/**
 * @brief The simulation time of one net signal.
 *
 */
class AnnotateTime {
 public:
  AnnotateTime(int64_t t0, int64_t t1, int64_t tx, int64_t tz)
      : _T0(t0), _T1(t1), _TX(tx), _TZ(tz) {}
  AnnotateTime() = default;
  ~AnnotateTime() = default;
  auto& get_T0() { return _T0; }
  void incrT0(int64_t duration) { _T0 += duration; }
  auto& get_T1() { return _T1; }
  void incrT1(int64_t duration) { _T1 += duration; }
  auto& get_TX() { return _TX; }
  void incrTX(int64_t duration) { _TX += duration; }
  auto& get_TZ() { return _TZ; }
  void incrTZ(int64_t duration) { _TZ += duration; }

  void printAnnotateTime(std::ostream& out);
  double get_SP() {
    double sp = (_T1.get_d() + _TZ.get_d()) /
                (_T1.get_d() + _T0.get_d() + _TX.get_d() + _TZ.get_d());
    return sp;
  }

 private:
  mpz_class _T0{0};  //!< The duration of zero for the signal.
  mpz_class _T1{0};  //!< The duration of one for the signal.
  mpz_class _TX{0};  //!< The duration of X for the signal.
  mpz_class _TZ{0};  //!< The duration of Z for the signal.
};

/**
 * @brief The simulation toggle of one net signal.
 *
 */
class AnnotateToggle {
 public:
  void incrTC() { ++_TC; }

  void printAnnotateToggle(std::ostream& out);
  int64_t get_toggle() { return _TC.get_ui(); }
  void set_TC(int tc) { _TC = tc; }

 private:
  mpz_class _TC{0};  //!< The total number of transition.
  mpz_class _TB{0};  //!< TBD
  mpz_class _TG{0};  //!< The number of glitch 0-1-0 and 1-0-1
  mpz_class _IG{0};  //!< The number of inertial glitch 0-x-0 and 1-x-1.
  mpz_class _IK{0};  //!< TBD
};

/**
 * @brief The simulation record of one signal.
 *
 */
class AnnotateRecord {
 public:
  AnnotateRecord(AnnotateToggle&& toggle_record, AnnotateTime&& time_record)
      : _toggle_record(std::move(toggle_record)),
        _time_record(std::move(time_record)) {}

  ~AnnotateRecord() = default;

  AnnotateRecord(AnnotateRecord&& other) noexcept = default;
  AnnotateRecord& operator=(AnnotateRecord&& other) noexcept = default;

  void set_toggle_record(AnnotateToggle&& toggle_record) {
    _toggle_record = std::move(toggle_record);
  }
  void set_time_record(AnnotateTime&& time_record) {
    _time_record = std::move(time_record);
  }

  void printAnnotateRecord(std::ostream& out) {
    _toggle_record.printAnnotateToggle(out);
    _time_record.printAnnotateTime(out);
  }

  std::pair<int64_t, double> get_record_tc_sp() {
    int64_t toggle = _toggle_record.get_toggle();
    double sp = _time_record.get_SP();
    return std::make_pair(toggle, sp);
  }

 private:
  AnnotateToggle _toggle_record;  //!< The toggle record of one net signal.
  AnnotateTime _time_record;      //!< The time record of one net signal.
};

/**
 * @brief The annotate signal.
 *
 */
class AnnotateSignal {
 public:
  explicit AnnotateSignal(const std::string& signal_name)
      : _signal_name(signal_name),
        _record_data(std::make_unique<AnnotateRecord>(AnnotateToggle(),
                                                      AnnotateTime())) {}
  ~AnnotateSignal() = default;

  auto& get_signal_name() { return _signal_name; }
  auto* get_record_data() { return _record_data.get(); }

  auto get_signal_tc_sp() { return _record_data->get_record_tc_sp(); }

  void printAnnotateSignal(std::ostream& out) {
    out << _signal_name << std::endl;
    _record_data->printAnnotateRecord(out);
  };

 private:
  std::string _signal_name;                      //!< The signal net name.
  std::unique_ptr<AnnotateRecord> _record_data;  //!< The signal record data.
};

/**
 * @brief the signal toggle and sp data.
 *
 */
class AnnotateSignalToggleSPData {
 public:
  AnnotateSignalToggleSPData(std::string_view signal_name, double toggle,
                             double sp)
      : _signal_name(signal_name), _toggle(toggle), _sp(sp){};
  ~AnnotateSignalToggleSPData() = default;
  auto& get_signal_name() { return _signal_name; }
  auto get_toggle() { return _toggle; }
  auto get_sp() { return _sp; }

 private:
  std::string_view _signal_name;
  double _toggle;
  double _sp;
};

/**
 * @brief The annotate instance.
 *
 */
class AnnotateInstance {
 public:
  explicit AnnotateInstance(const std::string& module_instance_name)
      : _module_instance_name(module_instance_name) {}
  ~AnnotateInstance() = default;

  auto& get_module_instance_name() { return _module_instance_name; }

  void addSignal(std::unique_ptr<AnnotateSignal> annotate_signal) {
    _signals[annotate_signal->get_signal_name()] = std::move(annotate_signal);
  }
  void addChildInstance(std::unique_ptr<AnnotateInstance> child) {
    std::string_view module_instance_name{child->get_module_instance_name()};
    _children_instances[module_instance_name] = std::move(child);
  }
  auto* findInstance(std::string_view child_instance_name) {
    LOG_FATAL_IF(!_children_instances.contains(child_instance_name))
        << child_instance_name << " is not found in the parent scope.";

    return _children_instances[child_instance_name].get();
  }
  auto* findSignal(std::string_view signal_name) {
    return _signals[signal_name].get();
  }

  void printAnnotateInstance(std::ostream& out);

  void calcInstancesTcSP(int64_t duration);
  auto& get_signals_tc_sp() { return _signals_tc_sp; }
  auto& get_children_instances() { return _children_instances; }
  auto& get_signals() { return _signals; }

 private:
  std::string _module_instance_name;
  std::map<std::string_view, std::unique_ptr<AnnotateSignal>>
      _signals;  //!< The signal of the instance.
  std::map<std::string_view, std::unique_ptr<AnnotateInstance>>
      _children_instances;  //!< The instance is hier, may have children.
  std::vector<std::unique_ptr<AnnotateSignalToggleSPData>>
      _signals_tc_sp;  //!< The signal toggle and sp data.
};

#define FOREACH_SIGNAL(instance, signal)                           \
  if (auto& signals = (instance)->get_signals(); !signals.empty()) \
    for (auto it = signals.begin();                                \
         it != signals.end() ? signal = it->second.get(), true : false; ++it)

#define FOREACH_CHILD_INSTANCE(parent_instance, instance)                   \
  if (auto& instances = (parent_instance)->get_children_instances();        \
      !instances.empty())                                                   \
    for (auto it = instances.begin();                                       \
         it != instances.end() ? instance = it->second.get(), true : false; \
         ++it)

/**
 * @brief Time scale of vcd simulation.
 *
 */
class AnnotateTimeScale {
 public:
  void set_annotate_time_scale(int64_t scale, int8_t unit_num) {
    _scale = scale;
    _unit = (ScaleUnit)unit_num;
  };

  std::string get_time_scale() {
    std::map<ScaleUnit, std::string_view> ScaleUnitStringMap = {
        {ScaleUnit::kSecond, "s"}, {ScaleUnit::kMS, "ms"},
        {ScaleUnit::kUS, "us"},    {ScaleUnit::kNS, "ns"},
        {ScaleUnit::kPS, "ps"},    {ScaleUnit::kFS, "fs"}};
    return (std::to_string(_scale.get_ui()) + " ")
        .std::string::append(ScaleUnitStringMap[_unit]);
  }

 private:
  mpz_class _scale;
  ScaleUnit _unit;
};

/**
 * @brief The annotate database.
 *
 */
class AnnotateDB {
 public:
  void set_top_instance(std::unique_ptr<AnnotateInstance> top_instance) {
    _top_instance = std::move(top_instance);
  }
  auto* get_top_instance() { return _top_instance.get(); }

  void set_simulation_start_time(int64_t simulation_start_time) {
    _simulation_start_time = simulation_start_time;
  }
  auto get_simulation_sart_time() { return _simulation_start_time.get_ui(); }

  void set_simulation_duration(int64_t simulation_duration) {
    _simulation_duration = simulation_duration;
  }
  auto get_simulation_duration() { return _simulation_duration.get_ui(); }

  auto getSimulationEndTime() {
    return _simulation_start_time.get_ui() + _simulation_duration.get_ui();
  }

  void set_timescale(int64_t scale, int8_t unit) {
    _timescale.set_annotate_time_scale(scale, unit);
  }
  auto get_timescale() { return _timescale; }

  void calcInstancesTcSP() {
    if (_top_instance) {
      _top_instance->calcInstancesTcSP(_simulation_duration.get_ui());
    }
  }

  auto& getTcSp() { return _top_instance->get_signals_tc_sp(); }

  AnnotateRecord* findSignalRecord(
      std::vector<std::string_view>& parent_instance_names,
      std::string_view signal_name);

  void printAnnotateDB(std::ostream& out);

 private:
  std::string _version;
  std::string _direction;
  std::string _date;
  std::string _vendor;
  std::string _program_name;
  std::string _tool_version;
  std::string _divider;

  mpz_class _simulation_start_time{0};
  mpz_class _simulation_duration{0};
  AnnotateTimeScale _timescale;

  std::unique_ptr<AnnotateInstance> _top_instance;
};

}  // namespace ipower
