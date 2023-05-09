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
#ifndef IDRC_SRC_DB_SPACING_RANGE_WIDTH_H_
#define IDRC_SRC_DB_SPACING_RANGE_WIDTH_H_
#include <algorithm>
#include <utility>
namespace idrc {
class SpacingRangeRule
{
 public:
  SpacingRangeRule() {}
  SpacingRangeRule(const SpacingRangeRule& other)
  {
    _min_width = other._min_width;
    _max_width = other._max_width;
    _spacing = other._spacing;
  }
  SpacingRangeRule(SpacingRangeRule&& other)
  {
    _min_width = std::move(other._min_width);
    _max_width = std::move(other._max_width);
    _spacing = std::move(other._spacing);
  }
  ~SpacingRangeRule() {}
  SpacingRangeRule& operator=(const SpacingRangeRule& other)
  {
    _min_width = other._min_width;
    _max_width = other._max_width;
    _spacing = other._spacing;
    return *this;
  }
  SpacingRangeRule& operator=(SpacingRangeRule&& other)
  {
    _min_width = std::move(other._min_width);
    _max_width = std::move(other._max_width);
    _spacing = std::move(other._spacing);
    return *this;
  }
  // setter
  void set_min_width(int min_width) { _min_width = min_width; }
  void set_max_width(int max_width) { _max_width = max_width; }
  void set_spacing(int spacing) { _spacing = spacing; }
  // getter
  int get_min_width() const { return _min_width; }
  int get_max_width() const { return _max_width; }
  int get_spacing() const { return _spacing; }

 private:
  int _min_width = 0;
  int _max_width = 0;
  int _spacing = 0;
};
}  // namespace idrc

#endif