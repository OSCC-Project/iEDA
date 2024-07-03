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
 * @file StaClock.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class of sta clock.
 * @version 0.1
 * @date 2021-02-17
 */
#pragma once

#include <utility>

#include "BTreeSet.hh"
#include "StaVertex.hh"
#include "Vector.hh"

namespace ista {

/**
 * @brief The class of clock waveform.
 *
 */
class StaWaveForm {
 public:
  StaWaveForm() = default;
  ~StaWaveForm() = default;

  StaWaveForm(StaWaveForm&& other) = default;
  StaWaveForm& operator=(StaWaveForm&& rhs) = default;

  void addWaveEdge(int wave_point) { _wave_edges.push_back(wave_point); }
  Vector<int>& get_wave_edges() { return _wave_edges; }

  int getRisingEdge() { return _wave_edges[0]; }
  int getFallingEdge() { return _wave_edges[1]; }

 private:
  Vector<int> _wave_edges;  //!< We assume that the edges compose of rising and
                            //!< falling edge pair.

  FORBIDDEN_COPY(StaWaveForm);
};

/**
 * @brief The clock for sta, which is convert from sdc clock.
 *
 */
class StaClock {
 public:
  enum class ClockType { kIdeal, kPropagated };

  StaClock(const char* clock_name, ClockType clock_type, int period);
  ~StaClock() = default;

  StaClock(StaClock&& other) = default;
  StaClock& operator=(StaClock&& rhs) = default;

  void addVertex(StaVertex* the_vertex) { _clock_vertexes.insert(the_vertex); }
  auto& get_clock_vertexes() { return _clock_vertexes; }

  const char* get_clock_name() { return _clock_name.c_str(); }

  void set_wave_form(StaWaveForm&& wave_form) {
    _wave_form = std::move(wave_form);
  }

  StaWaveForm& get_wave_form() { return _wave_form; }

  auto get_clock_type() { return _clock_type; }
  void set_clock_type(ClockType clock_type) { _clock_type = clock_type; }
  void setPropagateClock() { _clock_type = ClockType::kPropagated; }

  int getRisingEdge() { return _wave_form.getRisingEdge(); }
  int getFallingEdge() { return _wave_form.getFallingEdge(); }

  void set_is_need_update_period_waveform(bool is_true) {
    _is_need_update_period_waveform = is_true;
  }
  bool isNeedUpdatePeriodWaveform() const {
    return _is_need_update_period_waveform;
  }

  void set_period(int period_ps) { _period = period_ps; }
  [[nodiscard]] int get_period() const { return _period; }
  double getPeriodNs() const { return PS_TO_NS(_period); }

  [[nodiscard]] bool isIdealClockNetwork() const {
    return _clock_type == ClockType::kIdeal;
  }
  void set_ideal_clock_network_latency(double ideal_network_latency) {
    _ideal_network_latency = NS_TO_FS(ideal_network_latency);
  }

  auto& get_ideal_network_latency() { return _ideal_network_latency; }

  bool isSyncClockGroup(StaClock* other_clock) const { return false; }

  unsigned exec(StaFunc& func);

 private:
  std::string _clock_name;
  BTreeSet<StaVertex*>
      _clock_vertexes;  //!< The graph vertex which is clock point.
  ClockType _clock_type;

  std::optional<int> _ideal_network_latency;  //!< The clock network latency
                                              //!< is ideal, unit is ps.

  int _period;  // unit is ps.
  StaWaveForm _wave_form;

  bool _is_need_update_period_waveform =
      false;  //!< The flag of the time to clock prop.
  FORBIDDEN_COPY(StaClock);
};

}  // namespace ista
