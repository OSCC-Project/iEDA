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
#include "Module.hh"

#include <cstring>

namespace ipl::imp {

Module::Module()
{
  _name = "top";
  _layer = 0;
}

Module::~Module()
{
  for (Module* child_module : _child_module_list) {
    if (child_module != nullptr) {
      delete child_module;
      child_module = nullptr;
    }
    _child_module_list.clear();
  }

  _stdcell_list.clear();
}

void Module::add_inst(FPInst* inst, std::queue<std::string> level_name_list, std::string father_name)
{
  if (level_name_list.empty()) {
    std::string inst_name = inst->get_name();
    if ("" == inst_name) {
      return;
    }
    level_name_list = split(inst_name);
  }
  if (1 == level_name_list.size()) {
    if (inst->isMacro()) {
      _macro_list.emplace_back(inst);
    } else {
      _stdcell_list.emplace_back(inst);
    }
    return;
  }
  std::string level_name = level_name_list.front();
  level_name = father_name + "/" + level_name;
  Module* child_module = findChildMoudle(level_name);
  if (nullptr == child_module) {
    child_module = new Module();
    child_module->set_name(level_name);
    child_module->set_layer(_layer + 1);
    _child_module_list.emplace_back(child_module);
    _name_to_module_map.emplace(level_name, child_module);
  }
  level_name_list.pop();
  child_module->add_inst(inst, level_name_list, level_name);
}

Module* Module::findChildMoudle(std::string module_name)
{
  Module* child_module = nullptr;
  auto module_iter = _name_to_module_map.find(module_name);
  if (module_iter != _name_to_module_map.end()) {
    child_module = (*module_iter).second;
  }
  return child_module;
}

std::queue<std::string> Module::split(const string& str)
{
  std::queue<std::string> result;
  if ("" == str) {
    return result;
  }
  const char* delim = "/";
  char* strs = new char[str.length() + 1];
  strncpy(strs, str.c_str(), str.length() + 1);
  char* p = strtok(strs, delim);
  while (p) {
    std::string s = p;
    result.push(s);
    p = strtok(NULL, delim);
  }
  delete[] strs;
  return result;
}

}  // namespace ipl::imp