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

#include "DRCHeader.hpp"
#include "PlanarCoord.hpp"

namespace idrc {

class GPPath : public Segment<PlanarCoord>
{
 public:
  GPPath(const Segment<PlanarCoord>& segment, const int32_t layer_idx, const int32_t data_type, const int32_t width) : Segment<PlanarCoord>(segment)
  {
    _layer_idx = layer_idx;
    _data_type = data_type;
    _width = width;
  }
  GPPath() = default;
  ~GPPath() = default;
  // getter
  int32_t get_layer_idx() const { return _layer_idx; }
  int32_t get_data_type() const { return _data_type; }
  int32_t get_width() const { return _width; }
  Segment<PlanarCoord>& get_segment() { return (*this); }
  // setter
  void set_layer_idx(const int32_t layer_idx) { _layer_idx = layer_idx; }
  void set_data_type(const int32_t data_type) { _data_type = data_type; }
  void set_width(const int32_t width) { _width = width; }
  void set_segment(const Segment<PlanarCoord>& segment)
  {
    set_first(segment.get_first());
    set_second(segment.get_second());
  }
  void set_segment(const PlanarCoord& first_coord, const PlanarCoord& second_coord)
  {
    set_first(first_coord);
    set_second(second_coord);
  }
  void set_segment(const int32_t first_x, const int32_t first_y, const int32_t second_x, const int32_t second_y)
  {
    set_first(PlanarCoord(first_x, first_y));
    set_second(PlanarCoord(second_x, second_y));
  }
  // function

 private:
  int32_t _layer_idx = -1;
  int32_t _data_type = 0;
  int32_t _width = 1;
};

}  // namespace idrc
