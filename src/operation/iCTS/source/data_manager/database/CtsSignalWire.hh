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
 * @file CtsSignalWire.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once

#include <string>

#include "CtsPoint.hh"

namespace icts {

struct Endpoint
{
 public:
  std::string name;
  Point point;
};

class CtsSignalWire
{
 public:
  CtsSignalWire() = default;
  CtsSignalWire(const Endpoint& first, const Endpoint& second)
  {
    _wire.first = first;
    _wire.second = second;
  }
  CtsSignalWire(const CtsSignalWire&) = default;
  ~CtsSignalWire() = default;

  Endpoint get_first() const { return _wire.first; }
  Endpoint get_second() const { return _wire.second; }

  void set_first(const Endpoint& end_point) { _wire.first = end_point; }
  void set_second(const Endpoint& end_point) { _wire.second = end_point; }

 private:
  std::pair<Endpoint, Endpoint> _wire;
};

}  // namespace icts