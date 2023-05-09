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
 * @Date: 2022-02-23 20:43:10
 * @LastEditTime: 2022-03-04 11:06:01
 * @LastEditors: S.J Chen
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/database/Cell.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_CELL_H
#define IPL_CELL_H

#include <map>
#include <string>
#include <vector>

#include "Rectangle.hh"

namespace ipl {

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
  explicit Cell(std::string cell_name) : _cell_name(std::move(cell_name)), _cell_type(CELL_TYPE::kNone) {}
  Cell(const Cell& other) = delete;
  Cell(Cell&& other) = delete;
  ~Cell() = default;

  Cell& operator=(const Cell&) = delete;
  Cell& operator=(Cell&&) = delete;

  // getter.
  std::string get_name() const { return _cell_name; }
  int32_t get_width() const { return _width; }
  int32_t get_height() const { return _height; }
  CELL_TYPE get_cell_type() const { return _cell_type; }
  std::vector<std::string> get_inpin_name_list() const { return _inpin_name_list; }
  std::vector<std::string> get_outpin_name_list() const { return _outpin_name_list; }

  bool isLogic() { return _cell_type == CELL_TYPE::kLogic; }
  bool isFlipflop() { return _cell_type == CELL_TYPE::kFlipflop; }
  bool isClockBuffer() { return _cell_type == CELL_TYPE::kClockBuffer; }
  bool isLogicBuffer() { return _cell_type == CELL_TYPE::kLogicBuffer; }
  bool isMacro() { return _cell_type == CELL_TYPE::kMacro; }
  bool isIOCell() { return _cell_type == CELL_TYPE::kIOCell; }
  bool isPhysicalFiller() { return _cell_type == CELL_TYPE::kPhysicalFiller; }

  // setter.
  void set_type(CELL_TYPE cell_type) { _cell_type = cell_type; }
  void set_width(int32_t width) { _width = width; }
  void set_height(int32_t height) { _height = height; }
  void add_inpin_name(std::string inpin_name) { _inpin_name_list.push_back(inpin_name); }
  void add_outpin_name(std::string outpin_name) { _outpin_name_list.push_back(outpin_name); }

 private:
  std::string _cell_name;
  CELL_TYPE _cell_type;

  std::vector<std::string> _inpin_name_list;
  std::vector<std::string> _outpin_name_list;

  int32_t _width;
  int32_t _height;
};

}  // namespace ipl

#endif