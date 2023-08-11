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
#ifndef SRC_EVALUATOR_SOURCE_CONGESTION_DATABASE_CONGINST_HPP_
#define SRC_EVALUATOR_SOURCE_CONGESTION_DATABASE_CONGINST_HPP_

#include "CongPin.hpp"
#include "EvalRect.hpp"

namespace eval {

class CongInst
{
 public:
  CongInst() = default;
  ~CongInst() = default;

  // getter
  std::string get_name() const { return _name; }
  Rectangle<int64_t> get_shape() const { return _shape; }
  int64_t get_width() { return _shape.get_width(); }
  int64_t get_height() { return _shape.get_height(); }
  int64_t get_lx() const { return _shape.get_ll_x(); }
  int64_t get_ly() const { return _shape.get_ll_y(); }
  int64_t get_ux() const { return _shape.get_ur_x(); }
  int64_t get_uy() const { return _shape.get_ur_y(); }
  std::vector<CongPin*> get_pin_list() const { return _pin_list; }
  INSTANCE_LOC_TYPE get_loc_type() const { return _loc_type; }
  INSTANCE_STATUS get_status() const { return _status; }

  // booler
  bool isNormalInst() const { return _loc_type == INSTANCE_LOC_TYPE::kNormal; }
  bool isOutsideInst() const { return _loc_type == INSTANCE_LOC_TYPE::kOutside; }
  bool isFlipFlop() const { return _is_flip_flop; }

  // setter
  void set_name(const std::string& inst_name) { _name = inst_name; }
  void set_shape(const int64_t& lx, const int64_t& ly, const int64_t& ux, const int64_t& uy) { _shape.set_rectangle(lx, ly, ux, uy); }
  void set_pin_list(const std::vector<CongPin*>& cong_pin_list) { _pin_list = cong_pin_list; }
  void set_loc_type(const INSTANCE_LOC_TYPE& loc_type) { _loc_type = loc_type; }
  void set_flip_flop(const bool is_flip_flop) { _is_flip_flop = is_flip_flop; }
  void set_status(const INSTANCE_STATUS& status) { _status = status; }
  void add_pin(CongPin* pin) { _pin_list.push_back(pin); }

 private:
  std::string _name;
  Rectangle<int64_t> _shape;
  std::vector<CongPin*> _pin_list;
  INSTANCE_LOC_TYPE _loc_type;
  INSTANCE_STATUS _status;
  bool _is_flip_flop = false;
};

}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_CONGESTION_DATABASE_CONGINST_HPP_
