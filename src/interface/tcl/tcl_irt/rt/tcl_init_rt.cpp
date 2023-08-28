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
#include "RTAPI.hpp"
#include "tcl_rt.h"
#include "tcl_util.h"

namespace tcl {

// public

TclInitRT::TclInitRT(const char* cmd_name) : TclCmd(cmd_name)
{
  // std::string temp_directory_path;       // required
  _config_list.push_back(std::make_pair("-temp_directory_path", ValueType::kString));
  // irt_int log_level;                     // optional
  _config_list.push_back(std::make_pair("-log_level", ValueType::kInt));
  // irt_int thread_number;                 // optional
  _config_list.push_back(std::make_pair("-thread_number", ValueType::kInt));
  // std::string bottom_routing_layer;      // optional
  _config_list.push_back(std::make_pair("-bottom_routing_layer", ValueType::kString));
  // std::string top_routing_layer;         // optional
  _config_list.push_back(std::make_pair("-top_routing_layer", ValueType::kString));
  // irt_int enable_output_gds_files;       // optional
  _config_list.push_back(std::make_pair("-enable_output_gds_files", ValueType::kInt));
  // irt_int enable_idrc_interfaces;        // optional
  _config_list.push_back(std::make_pair("-enable_idrc_interfaces", ValueType::kInt));
  // irt_int pa_max_iter_num;               // optional
  _config_list.push_back(std::make_pair("-pa_max_iter_num", ValueType::kInt));
  // double ra_initial_penalty;             // optional
  _config_list.push_back(std::make_pair("-ra_initial_penalty", ValueType::kDouble));
  // double ra_penalty_drop_rate;           // optional
  _config_list.push_back(std::make_pair("-ra_penalty_drop_rate", ValueType::kDouble));
  // irt_int ra_outer_max_iter_num;         // optional
  _config_list.push_back(std::make_pair("-ra_outer_max_iter_num", ValueType::kInt));
  // irt_int ra_inner_max_iter_num;         // optional
  _config_list.push_back(std::make_pair("-ra_inner_max_iter_num", ValueType::kInt));
  // double gr_prefer_wire_unit;            // optional
  _config_list.push_back(std::make_pair("-gr_prefer_wire_unit", ValueType::kDouble));
  // double gr_via_unit;                    // optional
  _config_list.push_back(std::make_pair("-gr_via_unit", ValueType::kDouble));
  // double gr_corner_unit;                 // optional
  _config_list.push_back(std::make_pair("-gr_corner_unit", ValueType::kDouble));
  // double gr_access_history_cost_unit;    // optional
  _config_list.push_back(std::make_pair("-gr_access_history_cost_unit", ValueType::kDouble));
  // double gr_resource_history_cost_unit;  // optional
  _config_list.push_back(std::make_pair("-gr_resource_history_cost_unit", ValueType::kDouble));
  // irt_int gr_max_iter_num;               // optional
  _config_list.push_back(std::make_pair("-gr_max_iter_num", ValueType::kInt));
  // double ta_prefer_wire_unit;            // optional
  _config_list.push_back(std::make_pair("-ta_prefer_wire_unit", ValueType::kDouble));
  // double ta_nonprefer_wire_unit;         // optional
  _config_list.push_back(std::make_pair("-ta_nonprefer_wire_unit", ValueType::kDouble));
  // double ta_corner_unit;                 // optional
  _config_list.push_back(std::make_pair("-ta_corner_unit", ValueType::kDouble));
  // double ta_pin_distance_unit;           // optional
  _config_list.push_back(std::make_pair("-ta_pin_distance_unit", ValueType::kDouble));
  // double ta_group_distance_unit;         // optional
  _config_list.push_back(std::make_pair("-ta_group_distance_unit", ValueType::kDouble));
  // double ta_block_and_pin_unit;          // optional
  _config_list.push_back(std::make_pair("-ta_block_and_pin_unit", ValueType::kDouble));
  // double ta_reserved_via_unit;           // optional
  _config_list.push_back(std::make_pair("-ta_reserved_via_unit", ValueType::kDouble));
  // double ta_other_panel_unit;            // optional
  _config_list.push_back(std::make_pair("-ta_other_panel_unit", ValueType::kDouble));
  // double ta_self_panel_unit;             // optional
  _config_list.push_back(std::make_pair("-ta_self_panel_unit", ValueType::kDouble));
  // double ta_history_cost_unit;           // optional
  _config_list.push_back(std::make_pair("-ta_history_cost_unit", ValueType::kDouble));
  // irt_int ta_model_max_iter_num;         // optional
  _config_list.push_back(std::make_pair("-ta_model_max_iter_num", ValueType::kInt));
  // irt_int ta_panel_max_iter_num;         // optional
  _config_list.push_back(std::make_pair("-ta_panel_max_iter_num", ValueType::kInt));
  // double dr_block_and_pin_unit;          // optional
  _config_list.push_back(std::make_pair("-dr_block_and_pin_unit", ValueType::kDouble));
  // double dr_known_panel_unit;            // optional
  _config_list.push_back(std::make_pair("-dr_known_panel_unit", ValueType::kDouble));
  // double dr_reserved_via_unit;           // optional
  _config_list.push_back(std::make_pair("-dr_reserved_via_unit", ValueType::kDouble));
  // double dr_other_box_unit;              // optional
  _config_list.push_back(std::make_pair("-dr_other_box_unit", ValueType::kDouble));
  // double dr_self_box_unit;               // optional
  _config_list.push_back(std::make_pair("-dr_self_box_unit", ValueType::kDouble));
  // double dr_history_cost_unit;           // optional
  _config_list.push_back(std::make_pair("-dr_history_cost_unit", ValueType::kDouble));
  // irt_int dr_model_max_iter_num;         // optional
  _config_list.push_back(std::make_pair("-dr_model_max_iter_num", ValueType::kInt));
  // irt_int dr_box_max_iter_num;           // optional
  _config_list.push_back(std::make_pair("-dr_box_max_iter_num", ValueType::kInt));
  // irt_int vr_max_iter_num;               // optional
  _config_list.push_back(std::make_pair("-vr_max_iter_num", ValueType::kInt));

  TclUtil::addOption(this, _config_list);
}

unsigned TclInitRT::exec()
{
  std::map<std::string, std::any> config_map = TclUtil::getConfigMap(this, _config_list);
  RTAPI_INST.initRT(config_map);
  return 1;
}

// private

}  // namespace tcl
