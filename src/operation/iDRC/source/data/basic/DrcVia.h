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
#ifndef IDRC_SRC_DB_DRC_VIA_H_
#define IDRC_SRC_DB_DRC_VIA_H_

#include <string>
#include <utility>

#include "DrcCoordinate.h"
#include "DrcEnclosure.h"

namespace idrc {
class DrcVia
{
 public:
  explicit DrcVia() {}
  DrcVia(const DrcVia& other)
      : _via_idx(other._via_idx),
        _via_name(other._via_name),
        _center_coord(other._center_coord),
        _top_enclosure(other._top_enclosure),
        _bottom_enclosure(other._bottom_enclosure)
  {
  }
  DrcVia(DrcVia&& other)
  {
    _via_idx = std::move(other._via_idx);
    _via_name = std::move(other._via_name);
    _center_coord = std::move(other._center_coord);
    _top_enclosure = std::move(other._top_enclosure);
    _bottom_enclosure = std::move(other._bottom_enclosure);
  }
  ~DrcVia() {}
  DrcVia& operator=(const DrcVia& other)
  {
    _via_idx = other._via_idx;
    _via_name = other._via_name;
    _center_coord = other._center_coord;
    _top_enclosure = other._top_enclosure;
    _bottom_enclosure = other._bottom_enclosure;
    return (*this);
  }
  DrcVia& operator=(DrcVia&& other)
  {
    _via_idx = std::move(other._via_idx);
    _via_name = std::move(other._via_name);
    _center_coord = std::move(other._center_coord);
    _top_enclosure = std::move(other._top_enclosure);
    _bottom_enclosure = std::move(other._bottom_enclosure);
    return (*this);
  }

  // getter
  std::string& get_via_name() { return _via_name; }
  int get_via_idx() { return _via_idx; }
  DrcCoordinate<int>& get_center_coord() { return _center_coord; }
  DrcEnclosure& get_top_enclosure() { return _top_enclosure; }
  DrcEnclosure& get_bottom_enclosure() { return _bottom_enclosure; }
  // setter
  void set_via_idx(const int via_idx) { _via_idx = via_idx; }
  void set_via_name(const std::string& viaName) { _via_name = viaName; }
  void set_center_coord(const DrcCoordinate<int>& center_coord) { _center_coord = center_coord; }
  void set_top_enclosure(const DrcEnclosure& enclosure) { _top_enclosure = enclosure; }
  void set_bottom_enclosure(const DrcEnclosure& enclosure) { _bottom_enclosure = enclosure; }

  // function

 private:
  int _via_idx = -1;
  std::string _via_name;
  DrcCoordinate<int> _center_coord;
  DrcEnclosure _top_enclosure;
  DrcEnclosure _bottom_enclosure;
};
}  // namespace idrc

#endif