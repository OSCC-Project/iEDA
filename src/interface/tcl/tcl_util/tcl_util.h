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
