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
#include <iostream>
#include <string>
#include <vector>

#include "FPRect.hh"
#include "Coordinate.hh"
#include "Enum.hh"

namespace ipl::imp {

class FPPin;

class FPInst : public FPRect
{
 public:
  FPInst();
  ~FPInst();

  // setter
  void set_name(std::string name) { _name = name; }
  void set_index(int index) { _index = index; }
  void set_type(InstType type) { _type = type; }
  void set_fixed(bool fixed) { _fixed = fixed; }
  void set_halo_x(uint32_t halo) { _halo_x = halo; }
  void set_halo_y(uint32_t halo) { _halo_y = halo; }
  void set_orient(Orient orient);
  void add_pin(FPPin* pin) { _pin_list.emplace_back(pin); }
  void pop_pin_list() { _pin_list.pop_back(); }
  void set_align_flag(bool flag) { _align_flag = flag; }

  // getter
  std::string get_name() const { return _name; }
  int get_index() const { return _index; }
  uint32_t get_width() const override;
  uint32_t get_height() const override;
  uint32_t get_halo_x() const { return _halo_x; }
  uint32_t get_halo_y() const { return _halo_y; }
  InstType get_type() const { return _type; }
  Orient get_orient() const { return _orient; }
  // Different directions, different widths and heights
  int32_t get_center_x() const { return FPRect::get_x() + get_width() * 0.5; }
  int32_t get_center_y() const { return FPRect::get_y() + get_height() * 0.5; }
  std::vector<FPPin*> get_pin_list() const { return _pin_list; }
  bool isFixed() const { return _fixed; }
  void addHalo();
  void deleteHalo();
  bool isMacro() const { return _type == InstType::kMacro; }
  bool isNewMacro() const { return _type == InstType::kNew_macro; }
  std::string get_orient_name();
  bool isAlign() const { return _align_flag; }

 private:
  std::string _name;
  int _index;
  InstType _type;
  bool _fixed;
  uint32_t _halo_x;
  uint32_t _halo_y;
  bool _has_halo;
  Orient _orient;
  bool _main_orient;
  std::vector<FPPin*> _pin_list;
  bool _align_flag;
};
}  // namespace ipl::imp