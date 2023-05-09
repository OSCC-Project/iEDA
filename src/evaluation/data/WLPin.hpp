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
#ifndef SRC_PLATFORM_EVALUATOR_DATA_WLPIN_HPP_
#define SRC_PLATFORM_EVALUATOR_DATA_WLPIN_HPP_

#include "EvalPoint.hpp"
#include "EvalType.hpp"

namespace eval {

class WLPin
{
 public:
  WLPin() = default;
  ~WLPin() = default;

  // getter
  std::string get_name() const { return _name; }
  Point<int64_t> get_coord() const { return _coord; }
  int64_t get_x() const { return _coord.get_x(); }
  int64_t get_y() const { return _coord.get_y(); }
  PIN_TYPE get_type() const { return _type; }
  PIN_IO_TYPE get_io_type() const { return _io_type; }
  // int32_t get_layer_thickness() const { return _layer_thickness; }

  // setter
  void set_name(const std::string& name) { _name = name; }
  void set_coord(const Point<int64_t>& coord) { _coord = coord; }
  void set_x(const int64_t& x) { _coord.set_x(x); }
  void set_y(const int64_t& y) { _coord.set_y(y); }
  void set_type(const PIN_TYPE& type) { _type = type; }
  void set_io_type(const PIN_IO_TYPE& io_type) { _io_type = io_type; }
  // void set_layer_thickness(int32_t thickness) { _layer_thickness = thickness; }

  // pin type
  bool isIOPort() const { return _type == PIN_TYPE::kIOPort; }
  bool isIOInput() const { return (_type == PIN_TYPE::kIOPort) && (_io_type == PIN_IO_TYPE::kInput); }
  bool isIOOutput() const { return (_type == PIN_TYPE::kIOPort) && (_io_type == PIN_IO_TYPE::kOutput); }
  bool isIOInputOutput() const { return (_type == PIN_TYPE::kIOPort) && (_io_type == PIN_IO_TYPE::kInputOutput); }
  bool isInstancePort() const { return _type == PIN_TYPE::kInstancePort; }
  bool isInstanceInput() const { return (_type == PIN_TYPE::kInstancePort) && (_io_type == PIN_IO_TYPE::kInput); }
  bool isInstanceOutput() const { return (_type == PIN_TYPE::kInstancePort) && (_io_type == PIN_IO_TYPE::kOutput); }
  bool isInstanceInputOutput() const { return (_type == PIN_TYPE::kInstancePort) && (_io_type == PIN_IO_TYPE::kInputOutput); }

 private:
  std::string _name;
  Point<int64_t> _coord;
  PIN_TYPE _type;
  PIN_IO_TYPE _io_type;
  // int32_t _layer_thickness;
};

}  // namespace eval

#endif // SRC_PLATFORM_EVALUATOR_DATA_WLPIN_HPP_
