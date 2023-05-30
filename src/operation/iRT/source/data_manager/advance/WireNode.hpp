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

#include "PlanarCoord.hpp"
#include "Segment.hpp"

namespace irt {

class WireNode : public Segment<PlanarCoord>

{
 public:
  WireNode() = default;
  WireNode(const WireNode& other) : Segment<PlanarCoord>(other)
  {
    _net_idx = other._net_idx;
    _layer_idx = other._layer_idx;
    _wire_width = other._wire_width;
  }
  ~WireNode() = default;
  // getter
  irt_int get_net_idx() const { return _net_idx; }
  irt_int get_layer_idx() const { return _layer_idx; }
  irt_int get_wire_width() const { return _wire_width; }

  // setter
  void set_net_idx(const irt_int net_idx) { _net_idx = net_idx; }
  void set_layer_idx(const irt_int layer_idx) { _layer_idx = layer_idx; }
  void set_wire_width(const irt_int wire_width) { _wire_width = wire_width; }

  // function

 private:
  irt_int _net_idx = -1;
  irt_int _layer_idx = -1;
  irt_int _wire_width = -1;
};

}  // namespace irt
