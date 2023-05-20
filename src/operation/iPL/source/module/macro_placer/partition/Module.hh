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
#include <cstring>
#include <map>
#include <queue>
#include <string>
#include <vector>

#include "database/FPInst.hh"
namespace ipl::imp {

class Module
{
 public:
  Module();
  ~Module();
  void set_name(std::string name) { _name = name; }
  void add_inst(FPInst* inst, std::queue<std::string> level_name_list = {}, std::string father_name = "top");
  void set_layer(int layer) { _layer = layer; }

  std::string get_name() const { return _name; }
  std::vector<FPInst*> get_macro_list() const { return _macro_list; }
  std::vector<FPInst*> get_stdcell_list() const { return _stdcell_list; }
  std::vector<Module*> get_child_module_list() const { return _child_module_list; }
  bool hasChildModule() const { return _child_module_list.size() != 0; }
  int get_layer() const { return _layer; }
  Module* findChildMoudle(std::string module_name) const;

 private:
  std::queue<std::string> split(const std::string& str);
  std::string _name;
  int _layer;
  std::vector<FPInst*> _macro_list;
  std::vector<FPInst*> _stdcell_list;
  std::vector<Module*> _child_module_list;
  std::map<std::string, Module*> _name_to_module_map;
};

}  // namespace ipl::imp