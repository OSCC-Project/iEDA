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

#include <any>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "ScriptEngine.hh"
#include "json.hpp"

namespace tcl {

using ieda::ScriptEngine;
using ieda::TclCmd;
using ieda::TclCmds;
using ieda::TclDoubleListOption;
using ieda::TclDoubleOption;
using ieda::TclIntListOption;
using ieda::TclIntOption;
using ieda::TclOption;
using ieda::TclStringListOption;
using ieda::TclStringOption;
using ieda::TclSwitchOption;

enum class ValueType
{
  kInt,
  kIntList,
  kDouble,
  kDoubleList,
  kString,
  kStringList,
  kStringDoubleMap
};

class TclUtil : public TclCmd
{
 public:
  explicit TclUtil(const char* cmd_name) : TclCmd(cmd_name) {}
  ~TclUtil() override = default;
  unsigned check() override { return 1; };
  unsigned exec() override { return 1; };

  static void addOption(TclCmd* tcl_ptr, std::vector<std::pair<std::string, ValueType>> config_list);
  static void addOption(TclCmd* tcl_ptr, std::string config_name, ValueType type);
  static std::map<std::string, std::any> getConfigMap(TclCmd* tcl_ptr, std::vector<std::pair<std::string, ValueType>> config_list);
  static std::any getValue(TclCmd* tcl_ptr, std::string config_name, ValueType type);

 private:
  static std::vector<std::string> splitString(std::string a, char tok)
  {
    std::vector<std::string> result_list;
    while (true) {
      size_t pos = a.find(tok);
      if (std::string::npos == pos) {
        result_list.push_back(a);
        break;
      }
      result_list.push_back(a.substr(0, pos));
      a = a.substr(pos + 1);
    }
    return result_list;
  }
};

}  // namespace tcl
