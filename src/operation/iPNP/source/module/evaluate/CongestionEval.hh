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

// #include <iostream>
// #include <list>
// #include <string>
// #include <vector>

// using std::cout;
// using std::endl;
// using std::list;
// using std::string;
// using std::vector;

#include "iPNP.hh"

namespace ipnp {
class CongestionEval
{
 public:
  CongestionEval();
  ~CongestionEval();

  <type> getMapOverflow();
  double getCongValue() {return cong_value;};

 private:
  <type> cong_map_overflow;
  double cong_value;
  
};

}  // namespace ipnp
