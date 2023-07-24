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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include "FPInst.hh"
#include "FPPin.hh"

namespace ipl::imp {
class FPNet
{
 public:
  FPNet();  //_weight = 1.0;
  ~FPNet();

  void set_name(const std::string name) { _name = name; }
  void set_weight(float weight) { _weight = weight; }
  void add_inst(FPInst* inst) { _inst_set.insert(inst); }
  void add_pin(FPPin* pin);

  std::vector<FPPin*> get_pin_list() const { return _pin_list; }
  float get_weight() const { return _weight; }
  std::string get_name() const { return _name; }
  unsigned get_degree() const { return _pin_list.size(); }
  std::set<FPInst*> get_inst_set() const { return _inst_set; }

 private:
  std::string _name;
  float _weight;
  std::vector<FPPin*> _pin_list;
  std::set<FPInst*> _inst_set;
};
}  // namespace ipl::imp