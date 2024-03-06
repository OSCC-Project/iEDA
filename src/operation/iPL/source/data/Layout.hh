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
/*
 * @Author: S.J Chen
 * @Date: 2022-01-21 11:18:37
 * @LastEditTime: 2022-03-29 19:30:35
 * @LastEditors: S.J Chen
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/database/Layout.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_LAYOUT_H
#define IPL_LAYOUT_H

#include <map>
#include <string>

#include "Cell.hh"
#include "Rectangle.hh"
#include "Region.hh"
#include "Row.hh"

namespace ipl {

class Layout
{
 public:
  Layout() = default;
  Layout(const Layout&) = delete;
  Layout(Layout&&) = delete;
  ~Layout();

  Layout& operator=(const Layout&) = delete;
  Layout& operator=(Layout&&) = delete;

  // getter.
  int32_t get_database_unit() const { return _database_unit; }

  Rectangle<int32_t> get_die_shape() const { return _die_shape; }
  int32_t get_die_width() { return _die_shape.get_width(); }
  int32_t get_die_height() { return _die_shape.get_height(); }

  Rectangle<int32_t> get_core_shape() const { return _core_shape; }
  int32_t get_core_width() { return _core_shape.get_width(); }
  int32_t get_core_height() { return _core_shape.get_height(); }

  int32_t get_row_height() const { return _row_list[0]->get_site_height(); }
  int32_t get_site_width() const { return _row_list[0]->get_site_width(); }

  const std::vector<Row*> get_row_list() const { return _row_list; }
  const std::vector<Cell*> get_cell_list() const { return _cell_list; }
  int get_route_cap_h() const {return _route_cap_h;}
  int get_route_cap_v() const {return _route_cap_v;}
  int get_partial_route_cap_h() const {return _partial_route_cap_h;}
  int get_partial_route_cap_v() const {return _partial_route_cap_v;}

  // setter.
  void set_database_unit(int32_t dbu) { _database_unit = dbu; }
  void set_die_shape(Rectangle<int32_t> rect) { _die_shape = std::move(rect); }
  void set_core_shape(Rectangle<int32_t> rect) { _core_shape = std::move(rect); }
  void set_route_cap_h(int num) {_route_cap_h = num;}
  void set_route_cap_v(int num) {_route_cap_v = num;}
  void set_partial_route_cap_h(int num) {_partial_route_cap_h = num;}
  void set_partial_route_cap_v(int num) {_partial_route_cap_v = num;}

  void add_row(Row* row);
  void add_row_orient(Orient row_orient) { _row_orient_list.push_back(row_orient); }
  void add_cell(Cell* cell);

  // function.
  Row* find_row(const std::string& row_name) const;
  Row* find_row(int32_t row_id) const;
  Orient find_row_orient(int32_t row_index) const;
  Cell* find_cell(const std::string& cell_name) const;

 private:
  int32_t _database_unit = -1;
  Rectangle<int32_t> _die_shape;
  Rectangle<int32_t> _core_shape;

  std::vector<Row*> _row_list;
  std::vector<Orient> _row_orient_list;
  std::vector<Cell*> _cell_list;

  int _route_cap_h;
  int _route_cap_v;
  int _partial_route_cap_h;
  int _partial_route_cap_v;

  std::map<std::string, Row*> _name_to_row_map;
  std::map<std::string, Cell*> _name_to_cell_map;
};

inline Layout::~Layout()
{
  for (auto* row : _row_list) {
    delete row;
  }
  _row_list.clear();
  _name_to_row_map.clear();

  for (auto* cell : _cell_list) {
    delete cell;
  }
  _cell_list.clear();
  _name_to_cell_map.clear();
}

inline void Layout::add_row(Row* row)
{
  _row_list.push_back(row);
  _name_to_row_map.emplace(row->get_name(), row);
}

inline void Layout::add_cell(Cell* cell)
{
  _cell_list.push_back(cell);
  _name_to_cell_map.emplace(cell->get_name(), cell);
}

inline Row* Layout::find_row(const std::string& row_name) const
{
  Row* row = nullptr;
  auto row_iter = _name_to_row_map.find(row_name);
  if (row_iter != _name_to_row_map.end()) {
    row = (*row_iter).second;
  }
  return row;
}

inline Row* Layout::find_row(int32_t row_id) const
{
  return _row_list[row_id];
}

inline Orient Layout::find_row_orient(int32_t row_index) const
{
  return _row_orient_list[row_index];
}

inline Cell* Layout::find_cell(const std::string& cell_name) const
{
  Cell* cell = nullptr;
  auto cell_iter = _name_to_cell_map.find(cell_name);
  if (cell_iter != _name_to_cell_map.end()) {
    cell = (*cell_iter).second;
  }
  return cell;
}

}  // namespace ipl

#endif