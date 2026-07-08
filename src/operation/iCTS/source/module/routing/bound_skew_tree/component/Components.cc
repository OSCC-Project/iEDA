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
 * @file Components.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-23
 * @brief Implementation of TransformedRect methods that require Logger.
 */

#include "bound_skew_tree/component/Components.hh"

#include <glog/logging.h>

#include <algorithm>
#include <ostream>

#include "Log.hh"

namespace icts::bst {

auto TransformedRect::check() -> void
{
  correction();
  LOG_FATAL_IF(is_empty()) << "TRR is empty, which x_low: " << _x_low << ", x_high: " << _x_high << ", y_low: " << _y_low
                           << ", y_high: " << _y_high;
}

auto TransformedRect::correction() -> void
{
  auto temp_low = _x_low;
  auto temp_high = _x_high;
  if (Equal(temp_low, temp_high)) {
    _x_low = _x_high = (temp_low + temp_high) / 2;
  }
  temp_low = _y_low;
  temp_high = _y_high;
  if (Equal(temp_low, temp_high)) {
    _y_low = _y_high = (temp_low + temp_high) / 2;
  }
}

auto TransformedRect::intersect(const TransformedRect& trr1, const TransformedRect& trr2) -> TransformedRect
{
  auto x_low = std::max(trr1._x_low, trr2._x_low);
  auto x_high = std::min(trr1._x_high, trr2._x_high);
  auto y_low = std::max(trr1._y_low, trr2._y_low);
  auto y_high = std::min(trr1._y_high, trr2._y_high);
  auto trr = TransformedRect(x_low, x_high, y_low, y_high);
  trr.check();
  return trr;
}

}  // namespace icts::bst
