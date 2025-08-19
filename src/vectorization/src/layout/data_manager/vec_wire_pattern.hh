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
 * @file vec_wire_pattern.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-12-08
 * @brief wire pattern for vectorization
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "vec_net.h"

namespace ivec {
struct Point
{
  int x;
  int y;
  int z;
};
enum VecWirePatternDirection
{
  kTOP,
  kBOTTOM,
  kLEFT,
  kRIGHT,
  kVIA
};

struct VecWirePatternUnit
{
  VecWirePatternDirection direction;
  int length;
};

struct VecWirePatternSequence
{
  std::string name;
  std::vector<VecWirePatternUnit> units;
};

class VecWirePatternGenerator
{
 public:
  VecWirePatternGenerator() {}
  ~VecWirePatternGenerator() {}

  void genPatterns();
  void addPattern(VecNetWire& wire);
  void patternSummary(const std::string& csv_path);

 private:
  std::vector<Point> getPointList(VecNetWire& wire);
  VecWirePatternSequence calcPattern(const std::vector<Point>& points);

  std::unordered_map<std::string, VecWirePatternSequence> _patterns;
  std::unordered_map<std::string, int> _pattern_count;
};

}  // namespace ivec