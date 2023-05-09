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
#ifndef SRC_PLATFORM_EVALUATOR_SOURCE_GDS_WRITER_DATABASE_GDSWIRENODES_HPP_
#define SRC_PLATFORM_EVALUATOR_SOURCE_GDS_WRITER_DATABASE_GDSWIRENODES_HPP_

#include "EvalPoint.hpp"

namespace eval {

class GDSWireNodes
{
 public:
  GDSWireNodes() = default;
  ~GDSWireNodes() = default;

  int32_t get_layer_idx() const { return _layer_idx; }
  int32_t get_width() const { return _width; }
  Point<int32_t> get_first() const { return _first; }
  Point<int32_t> get_second() const { return _second; }

  void set_layer_idx(const int32_t& idx) { _layer_idx = idx; }
  void set_width(const int32_t& width) { _width = width; }
  void set_first(const Point<int32_t>& first) { _first = first; }
  void set_second(const Point<int32_t>& second) { _second = second; }

 private:
  int32_t _layer_idx = -1;
  int32_t _width = -1;
  Point<int32_t> _first;
  Point<int32_t> _second;
};
}  // namespace eval

#endif  // SRC_PLATFORM_EVALUATOR_SOURCE_GDS_WRITER_DATABASE_GDSWIRENODES_HPP_
