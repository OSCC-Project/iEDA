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
#ifndef SRC_PLATFORM_EVALUATOR_SOURCE_GDS_WRITER_DATABASE_GDSPORT_HPP_
#define SRC_PLATFORM_EVALUATOR_SOURCE_GDS_WRITER_DATABASE_GDSPORT_HPP_

#include <vector>

#include "EvalRect.hpp"

namespace eval {

class GDSPort
{
 public:
  GDSPort() = default;
  ~GDSPort() = default;

  int32_t get_layer_idx() const { return _layer_idx; }
  std::vector<Rectangle<int32_t>> get_shape_list() { return _shape_list; }

  void set_layer_idx(const int32_t& layer_idx) { _layer_idx = layer_idx; }
  void add_rect(const Rectangle<int32_t>& shape) { _shape_list.push_back(shape); }

 private:
  int32_t _layer_idx = -1;
  std::vector<Rectangle<int32_t>> _shape_list;
};

}  // namespace eval

#endif  // SRC_PLATFORM_EVALUATOR_SOURCE_GDS_WRITER_DATABASE_GDSPORT_HPP_
