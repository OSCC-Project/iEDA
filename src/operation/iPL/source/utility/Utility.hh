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
/*
 * @Author: S.J Chen
 * @Date: 2022-03-04 11:55:34
 * @LastEditTime: 2022-11-16 19:33:17
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/util/utility.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_UTIL_UTILITY_H
#define IPL_UTIL_UTILITY_H

#include <math.h>

#include <string>

#include "MultiTree.hh"
#include "data/Point.hh"

namespace ipl {

struct Utility
{
  Utility() = default;
  ~Utility() = default;

  int fastModulo(const int input, const int ceil);
  std::pair<int, int> obtainMinMaxIdx(int border_bottom, int interval, int inquiry_bottom, int inquiry_top);
  void correctPairRange(std::pair<int, int>& pair_range, int min, int32_t max);

  float getDistance(const std::vector<Point<int32_t>>& a, const std::vector<Point<int32_t>>& b);
  float getDistance(const std::vector<Point<float>>& a, const std::vector<Point<float>>& b);

  float calManhattanDistance(const Point<int32_t>& pos_1, const Point<int32_t>& pos_2);

  bool isFloatApproximatelyZero(float num);
  bool isFloatPairApproximatelyEqual(float num_1, float num_2);
};

inline int Utility::fastModulo(const int data, const int top)
{
  return data >= top ? data % top : data;
}

inline std::pair<int, int> Utility::obtainMinMaxIdx(int border_bottom, int interval, int inquiry_bottom, int inquiry_top)
{
  int lower_idx = (inquiry_bottom - border_bottom) / interval;
  int upper_idx = (fastModulo((inquiry_top - border_bottom), interval) == 0) ? (inquiry_top - border_bottom) / interval
                                                                             : (inquiry_top - border_bottom) / interval + 1;
  return std::make_pair(lower_idx, upper_idx);
}

inline void Utility::correctPairRange(std::pair<int, int>& pair_range, int min, int32_t max)
{
  if (pair_range.first < min) {
    pair_range.first = min;
  }
  if (pair_range.second > max) {
    pair_range.second = max;
  }
}

inline float Utility::getDistance(const std::vector<Point<int32_t>>& a, const std::vector<Point<int32_t>>& b)
{
  float sumDistance = 0.0f;
  for (size_t i = 0; i < a.size(); i++) {
    sumDistance += static_cast<float>(a[i].get_x() - b[i].get_x()) * static_cast<float>(a[i].get_x() - b[i].get_x());
    sumDistance += static_cast<float>(a[i].get_y() - b[i].get_y()) * static_cast<float>(a[i].get_y() - b[i].get_y());
  }

  // return std::sqrt(sumDistance / (2.0 * a.size()));
  return std::sqrt(sumDistance);
}

inline float Utility::getDistance(const std::vector<Point<float>>& a, const std::vector<Point<float>>& b)
{
  float sumDistance = 0.0f;
  for (size_t i = 0; i < a.size(); i++) {
    sumDistance += (a[i].get_x() - b[i].get_x()) * (a[i].get_x() - b[i].get_x());
    sumDistance += (a[i].get_y() - b[i].get_y()) * (a[i].get_y() - b[i].get_y());
  }

  // return std::sqrt(sumDistance / (2.0 * a.size()));
  return std::sqrt(sumDistance);
}

inline float Utility::calManhattanDistance(const Point<int32_t>& pos_1, const Point<int32_t>& pos_2)
{
  return (std::abs(pos_1.get_x() - pos_2.get_x()) + std::abs(pos_1.get_y() - pos_2.get_y()));
}

inline bool Utility::isFloatApproximatelyZero(float num)
{
  if (num < 1e-5) {
    return true;
  } else {
    return false;
  }
}

inline bool Utility::isFloatPairApproximatelyEqual(float num_1, float num_2)
{
  if (std::fabs(num_1 - num_2) < 1e-5) {
    return true;
  } else {
    return false;
  }
}

struct PointCMP
{
  bool operator()(const Point<int32_t>& lhs, const Point<int32_t>& rhs) const
  {
    if (lhs.get_x() == rhs.get_x()) {
      return lhs.get_y() < rhs.get_y();
    }

    return lhs.get_x() < rhs.get_x();
  }
};

}  // namespace ipl

#endif