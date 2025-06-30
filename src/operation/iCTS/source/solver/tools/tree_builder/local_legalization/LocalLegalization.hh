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
 * @file LocalLegalization.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once

#include <vector>

#include "Pin.hh"
namespace icts {
/**
 * @brief Local legalization only consider the input location but not the global instance and shape
 *
 */
class LocalLegalization
{
 public:
  LocalLegalization(Pin* driver_pin, const std::vector<Pin*>& load_pins);
  LocalLegalization(std::vector<Pin*>& pins);
  LocalLegalization(std::vector<Point>& variable_locations, const std::vector<Point>& fixed_locations = std::vector<Point>());

  ~LocalLegalization() = default;
  static void setIgnoreCore(const bool& ignore_core) { _ignore_core = ignore_core; }

 private:
  void legalize();
  static bool _ignore_core;
  std::vector<Point> _variable_locations;
  std::vector<Point> _fixed_locations;
};

}  // namespace icts