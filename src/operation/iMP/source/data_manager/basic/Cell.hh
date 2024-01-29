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

#ifndef IMP_CELL_H
#define IMP_CELL_H

#include <map>
#include <string>
#include <vector>

#include "Geometry.hh"

namespace imp {
enum class CELL_TYPE
{
  kNone,
  kLogic,
  kFlipflop,
  kClockBuffer,
  kLogicBuffer,
  kMacro,
  kIOCell,
  kPhysicalFiller
};
class Cell
{
 public:
  Cell() = delete;

  explicit Cell(std::string name, CELL_TYPE type = CELL_TYPE::kNone, geo::box<int32_t> box = geo::make_box(0, 0, 0, 0))
      : _name(name), _cell_type(type), _shape(box)
  {
  }
  Cell(const Cell& other) = delete;
  Cell(Cell&& other) = delete;
  ~Cell() = default;

  Cell& operator=(const Cell&) = delete;
  Cell& operator=(Cell&&) = delete;

  // getter.
  int32_t get_width() const { return geo::width(_shape); }
  int32_t get_height() const { return geo::height(_shape); }
  std::string get_name() const { return _name; }

  const geo::box<int32_t>& get_shape() const { return _shape; }
  geo::box<int32_t>& get_shape() { return _shape; }

  CELL_TYPE get_cell_type() const { return _cell_type; }

  bool isLogic() const { return _cell_type == CELL_TYPE::kLogic; }
  bool isFlipflop() const { return _cell_type == CELL_TYPE::kFlipflop; }
  bool isClockBuffer() const { return _cell_type == CELL_TYPE::kClockBuffer; }
  bool isLogicBuffer() const { return _cell_type == CELL_TYPE::kLogicBuffer; }
  bool isMacro() const { return _cell_type == CELL_TYPE::kMacro; }
  bool isIOCell() const { return _cell_type == CELL_TYPE::kIOCell; }
  bool isPhysicalFiller() const { return _cell_type == CELL_TYPE::kPhysicalFiller; }

  // setter.
  void set_type(CELL_TYPE cell_type) { _cell_type = cell_type; }
  void set_shape(const geo::box<int32_t>& shape) { _shape = shape; }

 private:
  std::string _name;
  CELL_TYPE _cell_type;

  geo::box<int32_t> _shape;
};

}  // namespace imp

#endif