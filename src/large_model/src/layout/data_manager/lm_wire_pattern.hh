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
 * @file lm_wire_pattern.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-12-08
 * @brief wire pattern for large model
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "lm_net.h"

namespace ilm {
struct Point
{
  int x;
  int y;
  int z;
};
enum LmWirePatternDirection
{
  kTOP,
  kBOTTOM,
  kLEFT,
  kRIGHT,
  kVIA
};

struct LmWirePatternUnit
{
  LmWirePatternDirection direction;
  int length;
};

struct LmWirePatternSequence
{
  std::string name;
  std::vector<LmWirePatternUnit> units;
};

class LmWirePatternGenerator
{
 public:
  LmWirePatternGenerator() = default;
  ~LmWirePatternGenerator() = default;

  void genPatterns();
  void addPattern(LmNetWire& wire);
  void patternSummary(const std::string& csv_path);

 private:
  std::vector<Point> getPointList(LmNetWire& wire);
  LmWirePatternSequence calcPattern(const std::vector<Point>& points);

  std::unordered_map<std::string, LmWirePatternSequence> _patterns;
  std::unordered_map<std::string, int> _pattern_count;
};

}  // namespace ilm