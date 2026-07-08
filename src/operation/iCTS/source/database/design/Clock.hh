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
 * @file Clock.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-07
 * @brief Main inputs and final membership view of one CTS clock
 */

#pragma once

#include <algorithm>
#include <string>
#include <vector>

namespace icts {

class Inst;
class Net;
class Pin;

class Clock
{
 public:
  Clock(const std::string& clock_name, const std::string& clock_net_name) : _clock_name(clock_name), _clock_net_name(clock_net_name) {}
  ~Clock() = default;

  // Getter
  auto get_clock_name() const -> const std::string& { return _clock_name; }
  auto get_clock_net_name() const -> const std::string& { return _clock_net_name; }
  auto get_clock_period_ns() const -> double { return _clock_period_ns; }
  auto get_clock_period_source() const -> const std::string& { return _clock_period_source; }
  auto get_clock_source() const -> Pin* { return _clock_source; }
  auto get_clock_source_net() const -> Net* { return _clock_source_net; }
  auto get_loads() const -> const std::vector<Pin*>& { return _loads; }
  auto get_insts() const -> const std::vector<Inst*>& { return _insts; }
  auto get_nets() const -> const std::vector<Net*>& { return _nets; }
  auto is_preclustered_sink_reuse() const -> bool { return _preclustered_sink_reuse; }
  auto get_preclustered_anchor_input_net_names() const -> const std::vector<std::string>& { return _preclustered_anchor_input_net_names; }

  // Setter
  auto set_clock_name(const std::string& clock_name) -> void { _clock_name = clock_name; }
  auto set_clock_net_name(const std::string& clock_net_name) -> void { _clock_net_name = clock_net_name; }
  auto set_clock_period_ns(double clock_period_ns) -> void { _clock_period_ns = clock_period_ns; }
  auto set_clock_period_source(const std::string& clock_period_source) -> void { _clock_period_source = clock_period_source; }
  auto set_clock_source(Pin* clock_source) -> void { _clock_source = clock_source; }
  auto set_clock_source_net(Net* clock_source_net) -> void { _clock_source_net = clock_source_net; }
  auto set_preclustered_sink_reuse(bool preclustered_sink_reuse) -> void { _preclustered_sink_reuse = preclustered_sink_reuse; }
  auto add_preclustered_anchor_input_net_name(const std::string& net_name) -> void
  {
    appendUnique(_preclustered_anchor_input_net_names, net_name);
  }
  auto clear_preclustered_anchor_input_net_names() -> void { _preclustered_anchor_input_net_names.clear(); }

  // Membership helpers.
  auto add_load(Pin* load) -> void { appendUnique(_loads, load); }
  auto set_loads(const std::vector<Pin*>& loads) -> void { _loads = loads; }
  auto clear_loads() -> void { _loads.clear(); }
  auto add_inst(Inst* inst) -> void { appendUnique(_insts, inst); }
  auto add_net(Net* net) -> void { appendUnique(_nets, net); }
  auto clearMembership() -> void
  {
    _insts.clear();
    _nets.clear();
  }

 private:
  static auto appendUnique(std::vector<std::string>& values, const std::string& value) -> void
  {
    if (value.empty() || std::ranges::find(values, value) != values.end()) {
      return;
    }
    values.push_back(value);
  }

  template <typename T>
  static auto appendUnique(std::vector<T*>& objects, T* object) -> void
  {
    if (object == nullptr || std::ranges::find(objects, object) != objects.end()) {
      return;
    }
    objects.push_back(object);
  }

  std::string _clock_name = "";
  std::string _clock_net_name = "";
  double _clock_period_ns = 0.0;
  std::string _clock_period_source = "";
  Pin* _clock_source = nullptr;
  Net* _clock_source_net = nullptr;
  std::vector<Pin*> _loads;
  std::vector<Inst*> _insts;
  std::vector<Net*> _nets;
  bool _preclustered_sink_reuse = false;
  std::vector<std::string> _preclustered_anchor_input_net_names;
};

}  // namespace icts
