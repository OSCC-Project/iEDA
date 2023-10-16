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
#include "Liberty.hh"
#include "log/Log.hh"

namespace ista {

/**
 * @brief Visit the liberty simple attribute statement.
 *
 * @param attri
 * @return unsigned return 1 if success, else 0
 */
unsigned RustLibertyReader::visitSimpleAttri(RustLibertySimpleAttrStmt* attri) {
  LibertyBuilder* lib_builder = get_library_builder();
  LibertyLibrary* current_lib = lib_builder->get_lib();
  LibertyPort* lib_port = lib_builder->get_port();
  if (!lib_port) {
    lib_port = lib_builder->get_port_bus();
  }
  LibertyCell* lib_cell = lib_builder->get_cell();
  LibertyLeakagePower* leakage_power = lib_builder->get_leakage_power();
  LibertyArc* lib_arc = lib_builder->get_arc();
  LibertyPowerArc* lib_power_arc = lib_builder->get_power_arc();
  auto* lib_obj = lib_builder->get_obj();
  LibertyBuilder::LibertyOwnPortType own_port_type =
      lib_builder->get_own_port_type();
  LibertyBuilder::LibertyOwnPgOrWhenType own_pg_or_when_type =
      lib_builder->get_own_pg_or_when_type();

  double cap_unit_convert = 1.0;  // sta use pf internal
  if (CapacitiveUnit::kFF == current_lib->get_cap_unit()) {
    cap_unit_convert = 0.001;
  }

  double resistance_unit_convert = 1000.0;  // sta use ohm internal
  if (ResistanceUnit::kOHM == current_lib->get_resistance_unit()) {
    cap_unit_convert = 1.0;
  }

  const char* attri_name = attri->attri_name;
  void* attri_value = const_cast<void*>(attri->attri_value);

  auto convert_string_to_bool = [](const std::string& str) -> bool {
    bool ret;
    std::istringstream(str) >> std::boolalpha >> ret;
    return ret;
  };

  std::map<std::string, std::function<void()>> process_attri = {
      {"nom_voltage",
       [=]() {
         double nom_voltage = rust_convert_float_value(attri_value)->value;
         current_lib->set_nom_voltage(nom_voltage);
       }},
      {"slew_lower_threshold_pct_rise",
       [=]() {
         double slew_lower_threshold_pct_rise =
             rust_convert_float_value(attri_value)->value;
         current_lib->set_slew_lower_threshold_pct_rise(
             slew_lower_threshold_pct_rise);
       }},
      {"slew_upper_threshold_pct_rise",
       [=]() {
         double slew_upper_threshold_pct_rise =
             rust_convert_float_value(attri_value)->value;
         current_lib->set_slew_upper_threshold_pct_rise(
             slew_upper_threshold_pct_rise);
       }},
      {"slew_lower_threshold_pct_fall",
       [=]() {
         double slew_lower_threshold_pct_fall =
             rust_convert_float_value(attri_value)->value;
         current_lib->set_slew_lower_threshold_pct_fall(
             slew_lower_threshold_pct_fall);
       }},
      {"slew_upper_threshold_pct_fall",
       [=]() {
         double slew_upper_threshold_pct_fall =
             rust_convert_float_value(attri_value)->value;
         current_lib->set_slew_upper_threshold_pct_fall(
             slew_upper_threshold_pct_fall);
       }},
      {"input_threshold_pct_rise",
       [=]() {
         double input_threshold_pct_rise =
             rust_convert_float_value(attri_value)->value;
         current_lib->set_input_threshold_pct_rise(input_threshold_pct_rise);
       }},
      {"output_threshold_pct_rise",
       [=]() {
         double output_threshold_pct_rise =
             rust_convert_float_value(attri_value)->value;
         current_lib->set_output_threshold_pct_rise(output_threshold_pct_rise);
       }},
      {"input_threshold_pct_fall",
       [=]() {
         double input_threshold_pct_fall =
             rust_convert_float_value(attri_value)->value;
         current_lib->set_input_threshold_pct_fall(input_threshold_pct_fall);
       }},
      {"output_threshold_pct_fall",
       [=]() {
         double output_threshold_pct_fall =
             rust_convert_float_value(attri_value)->value;
         current_lib->set_output_threshold_pct_fall(output_threshold_pct_fall);
       }},
      {"slew_derate_from_library",
       [=]() {
         double slew_derate_from_library =
             rust_convert_float_value(attri_value)->value;
         current_lib->set_output_threshold_pct_fall(slew_derate_from_library);
       }},
      {"pulling_resistance_unit",
       [=]() {
         const char* pulling_resistance_unit =
             rust_convert_string_value(attri_value)->value;
         if (Str::equal(pulling_resistance_unit, "1kohm")) {
           current_lib->set_resistance_unit(ResistanceUnit::kkOHM);
         }
       }},

      {"nom_voltage",
       [=]() {
         double nom_voltage = rust_convert_float_value(attri_value)->value;
         current_lib->set_nom_voltage(nom_voltage);
       }},

      {"default_max_transition",
       [=]() {
         double default_max_transition =
             rust_convert_float_value(attri_value)->value;
         current_lib->set_default_max_transition(default_max_transition);
       }},

      {"default_max_fanout",
       [=]() {
         double default_max_fanout =
             rust_convert_float_value(attri_value)->value;
         current_lib->set_default_max_fanout(default_max_fanout);
       }},
      {"direction",
       [=]() {
         const char* port_type = rust_convert_string_value(attri_value)->value;
         lib_port->set_port_type(port_type);
       }},
      {"clock_gate_clock_pin",
       [=]() {
         const char* clock_gate_clock_pin =
             rust_convert_string_value(attri_value)->value;
         bool clock_gate_clock_pin1 =
             convert_string_to_bool(clock_gate_clock_pin);
         lib_port->set_clock_gate_clock_pin(clock_gate_clock_pin1);
       }},
      {"clock_gate_enable_pin",
       [=]() {
         const char* clock_gate_enable_pin =
             rust_convert_string_value(attri_value)->value;
         bool clock_gate_enable_pin1 =
             convert_string_to_bool(clock_gate_enable_pin);
         lib_port->set_clock_gate_enable_pin(clock_gate_enable_pin1);
       }},
      {"default_fanout_load",
       [=]() {
         double default_fanout_load_val =
             rust_convert_float_value(attri_value)->value;
         current_lib->set_default_fanout_load(default_fanout_load_val);
       }},
      {"default_wire_load",
       [=]() {
         const char* default_wire_load =
             rust_convert_string_value(attri_value)->value;
         current_lib->set_default_wire_load(default_wire_load);
       }},
      {"fanout_load",
       [=]() {
         double fanout_load_val = rust_convert_float_value(attri_value)->value;
         lib_port->set_fanout_load(fanout_load_val);
       }},
      {"capacitance",
       [=]() {
         double cap = rust_convert_float_value(attri_value)->value;
         cap *= cap_unit_convert;
         if (lib_port) {
           lib_port->set_port_cap(cap);
         } else {
           dynamic_cast<LibertyWireLoad*>(lib_obj)->set_cap_per_length_unit(
               cap);
         }
       }},
      {"area",
       [=]() {
         double cell_area = rust_convert_float_value(attri_value)->value;
         if (lib_cell) {
           lib_cell->set_cell_area(cell_area);
         }
       }},
      {"is_macro_cell",
       [=]() {
         const char* is_macro = rust_convert_string_value(attri_value)->value;
         if (Str::noCaseEqual(is_macro, "TRUE")) {
           lib_cell->set_is_macro();
         }
       }},
      {"cell_leakage_power",
       [=]() {
         double cell_leakage_power =
             rust_convert_float_value(attri_value)->value;
         //  cell_leakage_power *= power_unit_convert;
         lib_cell->set_cell_leakage_power(cell_leakage_power);
       }},
      {"clock_gating_integrated_cell",
       [=]() {
         std::string clock_gating_integrated_cell =
             rust_convert_string_value(attri_value)->value;
         lib_cell->set_clock_gating_integrated_cell(
             clock_gating_integrated_cell);
         lib_cell->set_is_clock_gating_integrated_cell(true);
       }},
      {"resistance",
       [=]() {
         double resistance = rust_convert_float_value(attri_value)->value;
         resistance *= resistance_unit_convert;
         dynamic_cast<LibertyWireLoad*>(lib_obj)
             ->set_resistance_per_length_unit(resistance);
       }},
      {"slope",
       [=]() {
         double slope = rust_convert_float_value(attri_value)->value;
         dynamic_cast<LibertyWireLoad*>(lib_obj)->set_slope(slope);
       }},
      {"rise_capacitance",
       [=]() {
         double cap = rust_convert_float_value(attri_value)->value;
         cap *= cap_unit_convert;
         lib_port->set_port_cap(AnalysisMode::kMaxMin, TransType::kRise, cap);
       }},
      {"fall_capacitance",
       [=]() {
         double cap = rust_convert_float_value(attri_value)->value;
         cap *= cap_unit_convert;
         lib_port->set_port_cap(AnalysisMode::kMaxMin, TransType::kFall, cap);
       }},
      {"max_capacitance",
       [=]() {
         double max_cap_limit = rust_convert_float_value(attri_value)->value;
         max_cap_limit *= cap_unit_convert;
         lib_port->set_port_cap_limit(AnalysisMode::kMax, max_cap_limit);
       }},
      {"min_capacitance",
       [=]() {
         double min_cap_limit = rust_convert_float_value(attri_value)->value;
         min_cap_limit *= cap_unit_convert;
         lib_port->set_port_cap_limit(AnalysisMode::kMin, min_cap_limit);
       }},
      {"max_transition",
       [=]() {
         double max_slew_limit = rust_convert_float_value(attri_value)->value;
         lib_port->set_port_cap_limit(AnalysisMode::kMax, max_slew_limit);
       }},
      {"min_transition",
       [=]() {
         double min_slew_limit = rust_convert_float_value(attri_value)->value;
         lib_port->set_port_cap_limit(AnalysisMode::kMin, min_slew_limit);
       }},
      {"function",
       [=]() {
         const char* expr_str = rust_convert_string_value(attri_value)->value;
         LibertyExprBuilder expr_builder(lib_port, expr_str);
         expr_builder.execute();
         auto* func_expr = expr_builder.get_result_expr();
         lib_port->set_func_expr(func_expr);
         lib_port->set_func_expr_str(expr_str);
       }},
      {"related_pin",
       [=]() {
         const char* pin_name = rust_convert_string_value(attri_value)->value;
         if (own_port_type == LibertyBuilder::LibertyOwnPortType::kTimingArc) {
           lib_arc->set_src_port(pin_name);
         } else if (own_port_type ==
                    LibertyBuilder::LibertyOwnPortType::kPowerArc) {
           lib_power_arc->set_src_port(pin_name);
         }
       }},
      {"related_pg_pin",
       [=]() {
         const char* pg_pin_name =
             rust_convert_string_value(attri_value)->value;
         if (own_pg_or_when_type ==
             LibertyBuilder::LibertyOwnPgOrWhenType::kLibertyLeakagePower) {
           leakage_power->set_related_pg_port(pg_pin_name);
         } else if (own_pg_or_when_type ==
                    LibertyBuilder::LibertyOwnPgOrWhenType::kPowerArc) {
           lib_power_arc->set_related_pg_port(pg_pin_name);
         }
       }},
      {"when",
       [=]() {
         const char* when = rust_convert_string_value(attri_value)->value;
         if (own_pg_or_when_type ==
             LibertyBuilder::LibertyOwnPgOrWhenType::kLibertyLeakagePower) {
           leakage_power->set_when(when);
         } else if (own_pg_or_when_type ==
                    LibertyBuilder::LibertyOwnPgOrWhenType::kPowerArc) {
           if (lib_power_arc) {
             lib_power_arc->set_when(when);
           }
         }
       }},
      {"value",
       [=]() {
         if (rust_is_string_value(attri_value)) {
           const char* value =
               rust_convert_string_value(attri_value)->value;  // ysxy
           leakage_power->set_value(atof(value));
         } else {
           double value = rust_convert_float_value(attri_value)->value;  // T28
           leakage_power->set_value(value);
         }
       }},
      {"timing_sense",
       [=]() {
         const char* timing_sense =
             rust_convert_string_value(attri_value)->value;
         lib_arc->set_timing_sense(timing_sense);
       }},
      {"timing_type",
       [=]() {
         const char* timing_type =
             rust_convert_string_value(attri_value)->value;
         lib_arc->set_timing_type(timing_type);
       }},
      {"variable_1",
       [=]() {
         auto* lib_template = lib_obj;
         const char* variable_name =
             rust_convert_string_value(attri_value)->value;
         lib_template->set_template_variable1(variable_name);
       }},
      {"variable_2",
       [=]() {
         auto* lib_template = lib_obj;
         const char* variable_name =
             rust_convert_string_value(attri_value)->value;
         lib_template->set_template_variable2(variable_name);
       }},
      {"variable_3",
       [=]() {
         auto* lib_template = lib_obj;
         const char* variable_name =
             rust_convert_string_value(attri_value)->value;
         lib_template->set_template_variable3(variable_name);
       }},
      {"reference_time",
       [=]() {
         auto* lib_table = dynamic_cast<LibertyVectorTable*>(lib_obj);
         double ref_time = rust_convert_float_value(attri_value)->value;
         lib_table->set_ref_time(ref_time);
       }},
      {"base_type",
       [=]() {
         std::string base_type = rust_convert_string_value(attri_value)->value;
         dynamic_cast<LibertyType*>(lib_obj)->set_base_type(
             std::move(base_type));
       }},
      {"data_type",
       [=]() {
         std::string data_type = rust_convert_string_value(attri_value)->value;
         dynamic_cast<LibertyType*>(lib_obj)->set_data_type(
             std::move(data_type));
       }},
      {"bit_width",
       [=]() {
         double bit_width = rust_convert_float_value(attri_value)->value;
         dynamic_cast<LibertyType*>(lib_obj)->set_bit_width(
             static_cast<unsigned>(bit_width));
       }},
      {"bit_from",
       [=]() {
         double bit_from = rust_convert_float_value(attri_value)->value;
         dynamic_cast<LibertyType*>(lib_obj)->set_bit_from(
             static_cast<unsigned>(bit_from));
       }},
      {"bit_to",
       [=]() {
         double bit_to = rust_convert_float_value(attri_value)->value;
         dynamic_cast<LibertyType*>(lib_obj)->set_bit_to(
             static_cast<unsigned>(bit_to));
       }},
      {"bus_type", [=]() {
         auto* port_bus = lib_builder->get_port_bus();
         std::string bus_type = rust_convert_string_value(attri_value)->value;
         auto* lib_type = current_lib->getLibType(bus_type.c_str());
         port_bus->set_bus_type(lib_type);
       }}};

  if (process_attri.contains(attri_name)) {
    process_attri[attri_name]();
  }

  return 1;
}

/**
 * @brief Visit table axis and values.
 *
 * @param attri
 * @return unsigned
 */
unsigned RustLibertyReader::visitAxisOrValues(
    RustLibertyComplexAttrStmt* attri) {
  LibertyBuilder* lib_builder = get_library_builder();

  const char* attri_name = attri->attri_name;
  auto& attribute_values = attri->attri_values;

  /**
  @note the origial value may be quote by string.
   * So we need recover the double value.*/
  auto convert_attri_values = [](auto& attribute_values)
      -> std::vector<std::unique_ptr<LibertyAttrValue>> {
    auto split_str = [](std::string const& original,
                        char separator) -> std::vector<std::string> {
      std::vector<std::string> results;
      std::string token;
      std::istringstream is(original);
      while (std::getline(is, token, separator)) {
        if (!token.empty()) {
          results.push_back(token);
        }
      }
      return results;
    };

    std::vector<std::unique_ptr<LibertyAttrValue>> result_values;

    void* attri_value;
    FOREACH_VEC_ELEM(&attribute_values, void, attri_value) {
      if (rust_is_string_value(attri_value)) {
        std::string val = rust_convert_string_value(attri_value)->value;
        auto str_vec = split_str(val, ',');
        for (auto& str : str_vec) {
          auto double_val =
              std::make_unique<LibertyFloatValue>(std::atof(str.c_str()));
          result_values.emplace_back(std::move(double_val));
        }
      } else {
        double val = rust_convert_float_value(attri_value)->value;
        auto double_val = std::make_unique<LibertyFloatValue>(val);
        result_values.emplace_back(std::move(double_val));
      }
    }

    return result_values;
  };

  auto result_values = convert_attri_values(attribute_values);

  auto* lib_obj = lib_builder->get_obj();

  if (lib_obj) {
    if (Str::equal(attri_name, "values")) {
      auto* lib_table = dynamic_cast<LibertyTable*>(lib_obj);
      LOG_FATAL_IF(!lib_table);
      lib_table->set_table_values(std::move(result_values));
    } else {
      auto liberty_axis = std::make_unique<LibertyAxis>(attri_name);
      liberty_axis->set_axis_values(std::move(result_values));
      lib_obj->addAxis(std::move(liberty_axis));
    }
  }

  return 1;
}

/**
 * @brief Visit the liberty complex attribute statement.
 *
 * @param attri
 * @return unsigned return 1 if success, else 0
 */
unsigned RustLibertyReader::visitComplexAttri(
    RustLibertyComplexAttrStmt* attri) {
  const char* attri_name = attri->attri_name;
  LibertyBuilder* lib_builder = get_library_builder();
  auto* the_lib = lib_builder->get_lib();
  auto* lib_obj = lib_builder->get_obj();

  auto& attri_values = attri->attri_values;

  unsigned is_ok = 1;

  void* attri_0 = GetRustVecElem<void>(&attri_values, 0);
  void* attri_1 = GetRustVecElem<void>(&attri_values, 1);

  std::map<std::string, std::function<void()>> process_attri = {
      {"capacitive_load_unit",
       [&]() {
         if ((static_cast<int>(rust_convert_float_value(attri_0)->value) ==
              1) &&
             (Str::equal(rust_convert_string_value(attri_1)->value, "pf"))) {
           the_lib->set_cap_unit(CapacitiveUnit::kPF);
         }
       }},
      {"fanout_length", [&]() {
         double fanout = rust_convert_float_value(attri_0)->value;
         double length = rust_convert_float_value(attri_1)->value;
         dynamic_cast<LibertyWireLoad*>(lib_obj)->add_length_to_map(
             static_cast<int>(fanout), length);
       }}};

  if (process_attri.contains(attri_name)) {
    process_attri[attri_name]();
  } else {
    is_ok = visitAxisOrValues(attri);
  }
  return is_ok;
}

/**
 * @brief Visit group attri for name.
 *
 * @param group
 * @return const char*
 */
const char* RustLibertyReader::getGroupAttriName(RustLibertyGroupStmt* group) {
  auto& attri_values = group->attri_values;
  LOG_FATAL_IF(!rust_is_string_value(attri_values.data));
  auto* lib_name_attri = rust_convert_string_value(attri_values.data);

  return lib_name_attri->value;
}

/**
 * @brief Visit stmt of the group stmt.
 *
 * @param group
 * @return unsigned
 */
unsigned RustLibertyReader::visitStmtInGroup(RustLibertyGroupStmt* group) {
  unsigned is_ok = 1;

  auto lib_stmts = group->stmts;
  void* lib_stmt;
  FOREACH_VEC_ELEM(&lib_stmts, void, lib_stmt) {
    if (rust_is_simple_attri_stmt(lib_stmt)) {
      is_ok &= visitSimpleAttri(rust_convert_simple_attribute_stmt(lib_stmt));
    } else if (rust_is_complex_attri_stmt(lib_stmt)) {
      is_ok &= visitComplexAttri(rust_convert_complex_attribute_stmt(lib_stmt));
    } else {
      // group stmt.
      is_ok &= visitGroup(rust_convert_group_stmt(lib_stmt));
    }
  }

  return is_ok;
}

/**
 * @brief Visit library group stmt.
 *
 * @param group
 * @return unsigned
 */
unsigned RustLibertyReader::visitLibrary(RustLibertyGroupStmt* group) {
  const char* lib_name = getGroupAttriName(group);

  auto* library_builder = new LibertyBuilder(lib_name);
  set_library_builder(library_builder);

  unsigned is_ok = visitStmtInGroup(group);

  return 1;
}

/**
 * @brief Visit the wire load group stmt.
 *
 * @param group
 * @return unsigned
 */
unsigned RustLibertyReader::visitWireLoad(RustLibertyGroupStmt* group) {
  LibertyBuilder* lib_builder = get_library_builder();
  LibertyLibrary* lib = lib_builder->get_lib();

  const char* wire_load_name = getGroupAttriName(group);
  auto wire_load = std::make_unique<LibertyWireLoad>(wire_load_name);
  lib_builder->set_obj(wire_load.get());

  unsigned is_ok = visitStmtInGroup(group);

  lib->addWireLoad(std::move(wire_load));
  return is_ok;
}

/**
 * @brief Visit the lut table template group.
 *
 * @param group
 * @return unsigned
 */
unsigned RustLibertyReader::visitLuTableTemplate(RustLibertyGroupStmt* group) {
  LibertyBuilder* lib_builder = get_library_builder();
  LibertyLibrary* lib = lib_builder->get_lib();

  const char* template_name = getGroupAttriName(group);
  auto lut_table_template =
      std::make_unique<LibertyLutTableTemplate>(template_name);

  lib_builder->set_port(nullptr);
  lib_builder->set_obj(lut_table_template.get());

  unsigned is_ok = visitStmtInGroup(group);

  lib->addLutTemplate(std::move(lut_table_template));

  lib_builder->set_obj(nullptr);

  return is_ok;
}

/**
 * @brief Visit the type of lib.
 *
 * @param group
 * @return unsigned
 */
unsigned RustLibertyReader::visitType(RustLibertyGroupStmt* group) {
  LibertyBuilder* lib_builder = get_library_builder();
  LibertyLibrary* lib = lib_builder->get_lib();

  const char* type_name = getGroupAttriName(group);
  auto lib_type = std::make_unique<LibertyType>(type_name);

  lib_builder->set_port(nullptr);
  lib_builder->set_obj(lib_type.get());

  unsigned is_ok = visitStmtInGroup(group);

  lib->addLibType(std::move(lib_type));

  lib_builder->set_obj(nullptr);

  return is_ok;
}

/**
 * @brief Visit output current template.
 *
 * @param group
 * @return unsigned
 */
unsigned RustLibertyReader::visitOutputCurrentTemplate(
    RustLibertyGroupStmt* group) {
  LibertyBuilder* lib_builder = get_library_builder();
  LibertyLibrary* lib = lib_builder->get_lib();

  const char* template_name = getGroupAttriName(group);
  auto current_table_template =
      std::make_unique<LibertyCurrentTemplate>(template_name);

  lib_builder->set_obj(current_table_template.get());

  unsigned is_ok = visitStmtInGroup(group);

  lib->addLutTemplate(std::move(current_table_template));

  lib_builder->set_obj(nullptr);

  return is_ok;
}

/**
 * @brief Visit the cell group.
 *
 * @param group
 * @return unsigned return 1 if success, else 0
 */
unsigned RustLibertyReader::visitCell(RustLibertyGroupStmt* group) {
  LibertyBuilder* lib_builder = get_library_builder();
  LibertyLibrary* lib = lib_builder->get_lib();

  const char* cell_name = getGroupAttriName(group);

  auto lib_cell = std::make_unique<LibertyCell>(cell_name, lib);
  lib_builder->set_cell(lib_cell.get());

  unsigned is_ok = visitStmtInGroup(group);

  lib->addLibertyCell(std::move(lib_cell));

  return is_ok;
}

/**
 * @brief Visit leakage power.
 *
 * @param group
 * @return unsigned
 */
unsigned RustLibertyReader::visitLeakagePower(RustLibertyGroupStmt* group) {
  LibertyBuilder* lib_builder = get_library_builder();
  LibertyCell* lib_cell = lib_builder->get_cell();

  lib_builder->set_own_pg_or_when_type(
      LibertyBuilder::LibertyOwnPgOrWhenType::kLibertyLeakagePower);
  auto leakage_power = std::make_unique<LibertyLeakagePower>();
  lib_builder->set_leakage_power(leakage_power.get());
  leakage_power->set_owner_cell(lib_cell);

  unsigned is_ok = visitStmtInGroup(group);

  lib_cell->addLeakagePower(std::move(leakage_power));

  return is_ok;
}

/**
 * @brief Visit the bus pin.
 *
 * @param group
 * @return unsigned
 */
unsigned RustLibertyReader::visitBus(RustLibertyGroupStmt* group) {
  LibertyBuilder* lib_builder = get_library_builder();
  LibertyCell* cell = lib_builder->get_cell();

  const char* bus_port_name = getGroupAttriName(group);
  auto lib_port_bus = std::make_unique<LibertyPortBus>(bus_port_name);
  lib_port_bus->set_ower_cell(cell);
  lib_builder->set_port_bus(lib_port_bus.get());
  lib_builder->set_port(lib_port_bus.get());
  cell->addLibertyPortBus(std::move(lib_port_bus));

  unsigned is_ok = visitStmtInGroup(group);

  // reset the port bus pointer.
  lib_builder->set_port_bus(nullptr);

  return is_ok;
}

/**
 * @brief Visit the pin group.
 *
 * @param group
 * @return unsigned return 1 if success, else 0
 */
unsigned RustLibertyReader::visitPin(RustLibertyGroupStmt* group) {
  LibertyBuilder* lib_builder = get_library_builder();
  LibertyCell* cell = lib_builder->get_cell();

  const char* port_name = getGroupAttriName(group);

  auto create_port = [lib_builder, cell](const char* port_name) {
    auto lib_port = std::make_unique<LibertyPort>(port_name);
    lib_port->set_ower_cell(cell);

    if (auto* port_bus = lib_builder->get_port_bus(); !port_bus) {
      lib_builder->set_port(lib_port.get());
      cell->addLibertyPort(std::move(lib_port));
    } else {
      lib_port->set_port_type(port_bus->get_port_type());
      port_bus->addlibertyPort(std::move(lib_port));
    }
  };

  std::string regex_pattern = "([A-Za-z]+)\\[(\\d+):(\\d+)\\]";
  auto ret_val = Str::matchPattern(port_name, regex_pattern);
  if (ret_val.empty()) {
    create_port(port_name);
  } else {
    std::string port_bus_name = ret_val[1];
    int port_range_left = std::atoi(ret_val[2].c_str());
    int port_range_right = std::atoi(ret_val[3].c_str());

    for (int index = port_range_left; index >= port_range_right; --index) {
      const char* one_port_name =
          Str::printf("%s[%d]", port_bus_name.c_str(), index);
      create_port(one_port_name);
    }
  }

  unsigned is_ok = visitStmtInGroup(group);
  // reset the port pointer.
  lib_builder->set_port(nullptr);

  return is_ok;
}

/**
 * @brief Visit the timing group.
 *
 * @param group
 * @return unsigned return 1 if success, else 0
 */
unsigned RustLibertyReader::visitTiming(RustLibertyGroupStmt* group) {
  LibertyBuilder* lib_builder = get_library_builder();
  LibertyPort* lib_port = lib_builder->get_port();
  LibertyPortBus* lib_port_bus;
  if (!lib_port) {
    lib_port_bus = lib_builder->get_port_bus();
  }
  LibertyCell* lib_cell = lib_builder->get_cell();
  lib_builder->set_own_port_type(
      LibertyBuilder::LibertyOwnPortType::kTimingArc);
  auto lib_arc = std::make_unique<LibertyArc>();
  lib_builder->set_arc(lib_arc.get());
  lib_builder->set_table_model(nullptr);  // reset table model.
  lib_port ? lib_arc->set_snk_port(lib_port->get_port_name())
           : lib_arc->set_snk_port(lib_port_bus->get_port_name());
  lib_arc->set_owner_cell(lib_cell);

  unsigned is_ok = visitStmtInGroup(group);

  lib_cell->addLibertyArc(std::move(lib_arc));

  return is_ok;
}

/**
 * @brief Visit the internal power.
 *
 * @param group
 * @return unsigned
 */
unsigned RustLibertyReader::visitInternalPower(RustLibertyGroupStmt* group) {
  LibertyBuilder* lib_builder = get_library_builder();
  LibertyPort* lib_port = lib_builder->get_port();
  LibertyPortBus* lib_port_bus;
  if (!lib_port) {
    lib_port_bus = lib_builder->get_port_bus();
  }
  LibertyCell* lib_cell = lib_builder->get_cell();
  lib_builder->set_own_port_type(LibertyBuilder::LibertyOwnPortType::kPowerArc);
  lib_builder->set_own_pg_or_when_type(
      LibertyBuilder::LibertyOwnPgOrWhenType::kPowerArc);
  auto lib_power_arc = std::make_unique<LibertyPowerArc>();
  lib_builder->set_power_arc(lib_power_arc.get());
  lib_builder->set_table_model(nullptr);  // reset table model.
  lib_port ? lib_power_arc->set_snk_port(lib_port->get_port_name())
           : lib_power_arc->set_snk_port(lib_port_bus->get_port_name());
  lib_power_arc->set_owner_cell(lib_cell);

  auto internal_power_info = std::make_unique<LibertyInternalPowerInfo>();
  lib_power_arc->set_internal_power_info(std::move(internal_power_info));

  unsigned is_ok = 1;
  auto lib_stmts = group->stmts;
  void* lib_stmt;

  // for simple stmt, need first visit set powr arc attribute.
  FOREACH_VEC_ELEM(&lib_stmts, void, lib_stmt) {
    if (rust_is_simple_attri_stmt(lib_stmt)) {
      // for the power arc attribute.
      is_ok &= visitSimpleAttri(rust_convert_simple_attribute_stmt(lib_stmt));
    }
  }

  // visit group data finally.
  FOREACH_VEC_ELEM(&lib_stmts, void, lib_stmt) {
    if (rust_is_group_stmt(lib_stmt)) {
      // for the power data.
      // group stmt.
      is_ok &= visitGroup(rust_convert_group_stmt(lib_stmt));
    }
  }

  if (!lib_power_arc->isSrcPortEmpty()) {
    lib_cell->addLibertyPowerArc(std::move(lib_power_arc));
  } else {
    auto& internal_power_info = lib_power_arc->get_internal_power_info();
    lib_port ? lib_port->addInternalPower(std::move(internal_power_info))
             : lib_port_bus->addInternalPower(std::move(internal_power_info));
    lib_builder->set_power_arc(nullptr);
  }

  return is_ok;
}

/**
 * @brief Visit the output current table.
 *
 * @param group
 * @return unsigned
 */
unsigned RustLibertyReader::visitCurrentTable(RustLibertyGroupStmt* group) {
  LibertyBuilder* lib_builder = get_library_builder();
  auto* lib_model = lib_builder->get_table_model();
  auto* lib_delay_model = dynamic_cast<LibertyDelayTableModel*>(lib_model);

  const auto* const table_name = group->group_name;
  auto table_type = STR_TO_TABLE_TYPE(table_name);

  auto lib_table = std::make_unique<LibertyCCSTable>(table_type);
  lib_builder->set_current_table(lib_table.get());

  unsigned is_ok = visitStmtInGroup(group);

  lib_delay_model->addCurrentTable(std::move(lib_table));

  return is_ok;
}

/**
 * @brief Visit the current vector.
 *
 * @param group
 * @return unsigned
 */
unsigned RustLibertyReader::visitVector(RustLibertyGroupStmt* group) {
  LibertyBuilder* lib_builder = get_library_builder();

  const char* table_template_name = getGroupAttriName(group);
  auto* the_lib = lib_builder->get_lib();
  auto* lut_template = the_lib->getLutTemplate(table_template_name);
  LOG_FATAL_IF(!lut_template) << "not found template " << table_template_name;

  auto* current_table =
      dynamic_cast<LibertyCCSTable*>(lib_builder->get_current_table());
  auto table_type = current_table->get_table_type();

  auto lib_table =
      std::make_unique<LibertyVectorTable>(table_type, lut_template);
  lib_table->set_file_name(group->file_name);
  lib_table->set_line_no(group->line_no);

  lib_builder->set_obj(lib_table.get());

  current_table->addTable(std::move(lib_table));

  unsigned is_ok = visitStmtInGroup(group);

  return is_ok;
}

/**
 * @brief Visit the timing table group.
 *
 * @param group
 * @return unsigned
 */
unsigned RustLibertyReader::visitTable(RustLibertyGroupStmt* group) {
  LibertyBuilder* lib_builder = get_library_builder();

  const auto* const table_name = group->group_name;
  auto table_type = STR_TO_TABLE_TYPE(table_name);
  auto* lib_arc = lib_builder->get_arc();
  auto* lib_model = lib_builder->get_table_model();
  std::unique_ptr<LibertyTableModel> table_model;

  if (!lib_model) {
    if (lib_arc->isCheckArc()) {
      table_model = std::make_unique<LibertyCheckTableModel>();
    } else {
      table_model = std::make_unique<LibertyDelayTableModel>();
    }

    lib_builder->set_table_model(table_model.get());
    lib_model = lib_builder->get_table_model();
    lib_arc->set_table_model(std::move(table_model));
  }

  const char* table_template_name = getGroupAttriName(group);
  auto* the_lib = lib_builder->get_lib();
  auto* lut_template = the_lib->getLutTemplate(table_template_name);
  // LOG_FATAL_IF(!lut_template) << "not found template " <<
  // table_template_name;

  auto lib_table = std::make_unique<LibertyTable>(table_type, lut_template);
  lib_table->set_file_name(group->file_name);
  lib_table->set_line_no(group->line_no);

  lib_builder->set_table(lib_table.get());

  lib_model->addTable(std::move(lib_table));

  unsigned is_ok = visitStmtInGroup(group);
  return is_ok;
}

/**
 * @brief Visit the power table group.
 *
 * @param group
 * @return unsigned
 */
unsigned RustLibertyReader::visitPowerTable(RustLibertyGroupStmt* group) {
  LibertyBuilder* lib_builder = get_library_builder();

  const auto* const table_name = group->group_name;
  auto table_type = STR_TO_TABLE_TYPE(table_name);
  auto* lib_power_arc = lib_builder->get_power_arc();
  auto* lib_port = lib_builder->get_port();

  auto* lib_model = lib_builder->get_table_model();
  std::unique_ptr<LibertyTableModel> table_model;

  if (!lib_model) {
    table_model = std::make_unique<LibertyPowerTableModel>();

    lib_builder->set_table_model(table_model.get());
    lib_model = lib_builder->get_table_model();
    lib_power_arc->set_power_table_model(std::move(table_model));
  }

  const char* table_template_name = getGroupAttriName(group);
  auto* the_lib = lib_builder->get_lib();
  auto* lut_template = the_lib->getLutTemplate(table_template_name);
  // LOG_FATAL_IF(!lut_template) << "not found template " <<
  // table_template_name;

  auto lib_table = std::make_unique<LibertyTable>(table_type, lut_template);
  lib_table->set_file_name(group->file_name);
  lib_table->set_line_no(group->line_no);

  lib_builder->set_table(lib_table.get());

  // power_lib_model->addTable
  lib_model->addTable(std::move(lib_table));

  unsigned is_ok = visitStmtInGroup(group);

  return is_ok;
}

/**
 * @brief Visit the liberty group stmt to build liberty data.
 *
 * @param group
 * @return unsigned
 */
unsigned RustLibertyReader::visitGroup(RustLibertyGroupStmt* group) {
  DLOG_INFO << "visit group " << group->group_name << " line no "
            << group->line_no;
  if (group->line_no == 470) {
    LOG_INFO << "Debug";
  }
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
  } else {
    LOG_INFO << "group " << group_name << " is not supported.";
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
    auto* lib_group = rust_convert_raw_group_stmt(lib_file);
    unsigned result = visitGroup(lib_group);
    rust_free_lib_group(lib_file);

    LOG_INFO << "load liberty file " << _file_name << " success.";
    return result;
  }

  LOG_INFO << "load liberty file " << _file_name << " failed.";
  return 0;
}
}  // namespace ista