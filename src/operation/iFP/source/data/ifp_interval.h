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
#pragma once

#include <string>
#include <vector>

#include "ifp_enum.h"

namespace ifp {

class Interval
{
 public:
  Interval() = default;
  Interval(Edge edge, int begin, int end) : _edge(edge), _begin_position(begin), _end_position(end) {}
  ~Interval() = default;
  // get
  Edge get_edge() const { return _edge; }
  int32_t get_begin_position() const { return _begin_position; }
  int32_t get_end_position() const { return _end_position; }
  int32_t get_interval_length() const { return _end_position - _begin_position; }
  // set
  void set_edge(Edge edge) { _edge = edge; }
  void set_begin_position(int32_t begin) { _begin_position = begin; }
  void set_end_position(int32_t end) { _end_position = end; }

  // operator
  Interval& operator=(const Interval& interval)
  {
    if (this != &interval) {
      _edge = interval._edge;
      _begin_position = interval._begin_position;
      _end_position = interval._end_position;
    }
    return *this;
  }

 private:
  Edge _edge;
  int32_t _begin_position;
  int32_t _end_position;
};

}  // namespace ifp
