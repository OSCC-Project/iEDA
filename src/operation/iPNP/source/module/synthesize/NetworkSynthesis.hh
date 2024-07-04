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
/**
 * @brief Synthesize the whole PDN consisting of 3D Template and write DEF file. 
 * It doesn't include the function of deciding which template to place in which location. This is determined by the optimizer, or randomly, etc.
 */

#pragma once

#include <fstream>
#include <iostream>

#include "iPNP.hh"
#include "TemplateSynthesis.hh"
#include "DataManager.hh"

namespace ipnp {

class NetworkSynthesis
{
 public:
  NetworkSynthesis(std::string type, GridManager grid_info); //default is Random Synthesis.
  //NetworkSynthesis network(default, empty_grid_info);
  //NetworkSynthesis network(optimizer, optimizer_output_info);
  
  ~NetworkSynthesis();

  void synthesizeNetwork() { /* _synthesized_network = xxx */};

  void randomSys();

  //GridManager* getNetwork() { return &_synthesized_network; }  //返回值用指针还是类本身？
  GridManager &getNetwork() { return _synthesized_network; }  //返回值用指针还是类本身？

  file* writeDef(); //retrun type?
  //using iDB segment, refer to iPDN, iRT. SpecialNet?

 private:
  GridManager _input_grid_info; //用指针类型GridManager*还是类本身？
  GridManager _synthesized_network; //the whole PDN consist of Templates
  std::string _nework_sys_type; //{default, best, worst, optimizer}

};

}  // namespace ipnp