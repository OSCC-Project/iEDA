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
 * @file SdcException.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The class of sdc exception, such as set_false_path,
 * set_max_delay/set_min_delay, set_multicycle_path.
 * @version 0.1
 * @date 2022-07-18
 */
#pragma once

#include <utility>

#include "SdcCommand.hh"
#include "netlist/Netlist.hh"

namespace ista {

/**
 * @brief The sdc exception obj.
 *
 */
class SdcException : public SdcCommandObj {
 public:
  using ExceptionList = std::vector<std::string>;
  void set_prop_froms(ExceptionList&& prop_froms) {
    _prop_froms = std::move(prop_froms);
  }
  auto& get_prop_froms() { return _prop_froms; }

  void set_prop_tos(ExceptionList&& prop_tos) {
    _prop_tos = std::move(prop_tos);
  }
  auto& get_prop_tos() { return _prop_tos; }

  void set_prop_throughs(std::vector<ExceptionList>&& prop_throughs) {
    _prop_throughs = std::move(prop_throughs);
  }
  auto& get_prop_throughs() { return _prop_throughs; }

  virtual unsigned isFalsePath() { return 0; }
  virtual unsigned isMulticyclePath() { return 0; }
  virtual unsigned isMaxMinDelay() { return 0; }

 private:
  ExceptionList _prop_froms;
  ExceptionList _prop_tos;
  std::vector<ExceptionList> _prop_throughs;
};

/**
 * @brief The set_multicycle_path exception.
 *
 */
class SdcMulticyclePath : public SdcException {
 public:
  explicit SdcMulticyclePath(int path_multiplier);
  ~SdcMulticyclePath() override = default;

  void set_setup(bool is_set) { _setup = is_set; }
  void set_hold(bool is_set) { _hold = is_set; }

  [[nodiscard]] unsigned isSetup() const { return _setup; }
  [[nodiscard]] unsigned isHold() const { return _hold; }

  void set_rise(bool is_set) { _rise = is_set; }
  void set_fall(bool is_set) { _fall = is_set; }

  [[nodiscard]] unsigned isRise() const { return _rise; }
  [[nodiscard]] unsigned isFall() const { return _fall; }

  void set_start(bool is_set) { _start = is_set; }
  void set_end(bool is_set) { _end = is_set; }

  [[nodiscard]] unsigned isStart() const { return _start; }
  [[nodiscard]] unsigned isEnd() const { return _end; }

  [[nodiscard]] int get_path_multiplier() const { return _path_multiplier; }

  unsigned isMulticyclePath() override { return 1; }

 private:
  int _path_multiplier;
  unsigned _setup : 1 = 1;
  unsigned _hold : 1 = 1;
  unsigned _rise : 1 = 1;
  unsigned _fall : 1 = 1;
  unsigned _start : 1 = 1;
  unsigned _end : 1 = 1;
  unsigned _reserved : 26;
};

}  // namespace ista