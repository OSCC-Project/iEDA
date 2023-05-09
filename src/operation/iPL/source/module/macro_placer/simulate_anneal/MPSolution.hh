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
#include <string>
#include <vector>

#include "Solution.hh"
#include "database/FPInst.hh"

namespace ipl::imp {
class MPSolution : public Solution
{
 public:
  MPSolution(vector<FPInst*> macro_list)
  {
    _num_macro = macro_list.size();
    _macro_list = macro_list;
  }
  uint32_t get_total_width() { return _total_width; }
  uint32_t get_total_height() { return _total_height; }
  float get_total_area() { return _total_area; }
  virtual void printSolution(){};

 protected:
  int _num_macro = 0;
  vector<FPInst*> _macro_list;
  uint32_t _total_width = 0;
  uint32_t _total_height = 0;
  float _total_area = 0;
};

}  // namespace ipl::imp