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
#include "GPTextPresentation.hpp"
#include "PlanarCoord.hpp"

namespace idrc {

class GPText
{
 public:
  GPText(int32_t layer_idx, int32_t text_type, GPTextPresentation presentation, PlanarCoord coord, std::string message)
      : _layer_idx(layer_idx), _text_type(text_type), _presentation(presentation), _coord(coord), _message(message)
  {
  }
  GPText() = default;
  ~GPText() = default;
  // getter
  int32_t get_layer_idx() const { return _layer_idx; }
  int32_t get_text_type() const { return _text_type; }
  GPTextPresentation& get_presentation() { return _presentation; }
  PlanarCoord& get_coord() { return _coord; }
  std::string& get_message() { return _message; }

  // setter
  void set_layer_idx(const int32_t layer_idx) { _layer_idx = layer_idx; }
  void set_text_type(const int32_t text_type) { _text_type = text_type; }
  void set_presentation(const GPTextPresentation& presentation) { _presentation = presentation; }
  void set_coord(const PlanarCoord& coord) { _coord = coord; }
  void set_coord(const int32_t x, const int32_t y) { _coord.set_coord(x, y); }
  void set_message(const std::string& message) { _message = message; }
  // function

 private:
  int32_t _layer_idx;
  int32_t _text_type = 0;
  GPTextPresentation _presentation = GPTextPresentation::kNone;
  PlanarCoord _coord;
  std::string _message;
};

}  // namespace idrc
