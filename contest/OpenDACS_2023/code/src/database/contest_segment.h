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

#include "contest_coord.h"

namespace ieda_contest {

class ContestSegment
{
 public:
  ContestSegment() = default;
  ContestSegment(const ContestCoord& first, const ContestCoord& second)
  {
    _first = first;
    _second = second;
  }
  ~ContestSegment() = default;
  // getter
  ContestCoord& get_first() { return _first; }
  ContestCoord& get_second() { return _second; }
  // setter
  void set_first(const ContestCoord& first) { _first = first; }
  void set_second(const ContestCoord& second) { _second = second; }

 private:
  ContestCoord _first;
  ContestCoord _second;
};

struct SortSegmentInner
{
  void operator()(ContestSegment& a) const
  {
    ContestCoord& first_coord = a.get_first();
    ContestCoord& second_coord = a.get_second();
    if (CmpContestCoord()(first_coord, second_coord)) {
      return;
    }
    std::swap(first_coord, second_coord);
  }
};

struct CmpSegment
{
  bool operator()(ContestSegment& a, ContestSegment& b) const
  {
    if (a.get_first() != b.get_first()) {
      return CmpContestCoord()(a.get_first(), b.get_first());
    } else {
      return CmpContestCoord()(a.get_second(), b.get_second());
    }
  }
};

}  // namespace ieda_contest
