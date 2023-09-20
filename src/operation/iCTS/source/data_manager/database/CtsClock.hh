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
 * @file CtsClock.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once

#include <string>
#include <vector>

#include "CtsNet.hh"

namespace icts {

class CtsClock
{
 public:
  CtsClock() = default;
  CtsClock(const std::string& clock_name) : _clock_name(clock_name) {}
  CtsClock(const CtsClock&) = default;
  ~CtsClock() = default;

  // getter
  std::string get_clock_name() const { return _clock_name; }
  std::vector<CtsNet*>& get_clock_nets() { return _clock_nets; }

  // setter
  void set_clock_name(const std::string& clock_name) { _clock_name = clock_name; }

  void addClockNet(CtsNet* net) { _clock_nets.push_back(net); }

 private:
  std::string _clock_name;
  std::vector<CtsNet*> _clock_nets;
};
}  // namespace icts