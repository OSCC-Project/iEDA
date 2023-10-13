// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file LibertyParserRustC.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The libertyParser Rust C API.
 * @version 0.1
 * @date 2023-10-13
 *
 */
#include "LibertyParserRustC.hh"

#include <map>

#include "BTreeMap.hh"
#include "BTreeSet.hh"
#include "log/Log.hh"

namespace ista {

/**
 * @brief visit library group stmt.
 *
 * @param group
 * @return unsigned
 */
unsigned RustLibertyReader::visitLibrary(RustLibertyGroupStmt* group) {
  auto& attri_values = group->attri_values;
  LOG_FATAL_IF(!rust_is_string_value(attri_values.data));
  auto* lib_name_attri = rust_convert_string_value(attri_values.data);
  const char* lib_name = lib_name_attri->value;

  auto lib_stmts = group->stmts;
  void* lib_stmt;
  FOREACH_VEC_ELEM(&lib_stmts, void, lib_stmt) {
    if (rust_is_simple_attri_stmt(lib_stmt)) {
    } else if (rust_is_complex_attri_stmt(lib_stmt)) {
    } else {
      // group stmt.
    }
  }

  return 1;
}

/**
 * @brief Visit the liberty group stmt to build liberty data.
 *
 * @param group
 * @return unsigned
 */
unsigned RustLibertyReader::visitGroup(RustLibertyGroupStmt* group) {
  DLOG_INFO << "visit group " << group->group_name;
  unsigned is_ok = 1;
  const char* group_name = group->group_name;

  static const ieda::BTreeSet<std::string> table_names = {
      "cell_rise",       "cell_fall",       "rise_transition",
      "fall_transition", "rise_constraint", "fall_constraint"};
  static const ieda::BTreeSet<std::string> power_table_names = {"rise_power",
                                                                "fall_power"};

  using std::placeholders::_1;

  std::map<std::string, std::function<unsigned(RustLibertyGroupStmt * group)>>
      visit_fun_map = {
          {"library", std::bind(&RustLibertyReader::visitLibrary, this, _1)},
          {"wire_load", std::bind(&RustLibertyReader::visitWireLoad, this, _1)},
          {"lu_table_template",
           std::bind(&RustLibertyReader::visitLuTableTemplate, this, _1)},
          {"power_lut_template",
           std::bind(&RustLibertyReader::visitLuTableTemplate, this, _1)},
          {"type", std::bind(&RustLibertyReader::visitType, this, _1)},
          {"output_current_template",
           std::bind(&RustLibertyReader::visitOutputCurrentTemplate, this, _1)},
          {"cell", std::bind(&RustLibertyReader::visitCell, this, _1)},
          {"leakage_power",
           std::bind(&RustLibertyReader::visitLeakagePower, this, _1)},
          {"bus", std::bind(&RustLibertyReader::visitBus, this, _1)},
          {"pin", std::bind(&RustLibertyReader::visitPin, this, _1)},
          {"timing", std::bind(&RustLibertyReader::visitTiming, this, _1)},
          {"internal_power",
           std::bind(&RustLibertyReader::visitInternalPower, this, _1)},
          {"output_current_rise",
           std::bind(&RustLibertyReader::visitCurrentTable, this, _1)},
          {"output_current_fall",
           std::bind(&RustLibertyReader::visitCurrentTable, this, _1)},
          {"vector", std::bind(&RustLibertyReader::visitVector, this, _1)}};

  if (visit_fun_map.contains(group_name)) {
    auto read_func = visit_fun_map[group_name];
    is_ok = read_func(group);
  } else if (table_names.contains(group_name)) {
    is_ok = visitTable(group);
  } else if (power_table_names.contains(group_name)) {
    is_ok = visitPowerTable(group);
  }

  return 1;
}

/**
 * @brief Read the lib file use rust parser.
 *
 * @return unsigned
 */
unsigned RustLibertyReader::readLib() {
  LOG_INFO << "load liberty file " << _file_name;
  auto* lib_file = rust_parse_lib(_file_name.c_str());
  if (lib_file) {
    auto* lib_group = rust_convert_group_stmt(lib_file);
    unsigned result = visitGroup(lib_group);

    LOG_INFO << "load liberty file " << _file_name << " success.";
    return result;
  }

  LOG_INFO << "load liberty file " << _file_name << " failed.";
  return 0;
}
}  // namespace ista