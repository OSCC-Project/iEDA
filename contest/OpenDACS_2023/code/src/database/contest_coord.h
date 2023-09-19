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
 * @File Name: contest_guide.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2023-09-15
 *
 */
#pragma once
#include <set>
#include <string>
#include <vector>

namespace ieda_contest {

class ContestCoord
{
 public:
  ContestCoord() = default;
  ContestCoord(int x, int y, int layer_idx = -1)
  {
    _x = x;
    _y = y;
    _layer_idx = layer_idx;
  }
  ~ContestCoord() = default;
  bool operator==(const ContestCoord& other) const { return (_x == other._x && _y == other._y && _layer_idx == other._layer_idx); }
  bool operator!=(const ContestCoord& other) const { return !((*this) == other); }
  // getter
  int get_x() const { return _x; }
  int get_y() const { return _y; }
  int get_layer_idx() const { return _layer_idx; }
  // setter
  void set_x(const int x) { _x = x; }
  void set_y(const int y) { _y = y; }
  void set_layer_idx(const int layer_idx) { _layer_idx = layer_idx; }

 private:
  int _x = -1;
  int _y = -1;
  int _layer_idx = -1;
};

struct CmpContestCoord
{
  bool operator()(const ContestCoord& a, const ContestCoord& b) const
  {
    if (a.get_x() != b.get_x()) {
      return a.get_x() < b.get_x();
    } else {
      return a.get_y() != b.get_y() ? a.get_y() < b.get_y() : a.get_layer_idx() < b.get_layer_idx();
    }
  }
};

}  // namespace ieda_contest
