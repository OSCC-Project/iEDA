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

#include "RTU.hpp"
#include "Stage.hpp"

namespace irt {

class Config
{
 public:
  Config() = default;
  ~Config() = default;
  /////////////////////////////////////////////
  // **********        RT         ********** //
  std::string temp_directory_path;       // required
  irt_int log_level;                     // optional
  irt_int thread_number;                 // optional
  std::string bottom_routing_layer;      // optional
  std::string top_routing_layer;         // optional
  irt_int enable_output_gds_files;       // optional
  irt_int enable_idrc_interfaces;        // optional
  irt_int pa_max_iter_num;               // optional
  double ra_initial_penalty;             // optional
  double ra_penalty_drop_rate;           // optional
  irt_int ra_outer_max_iter_num;         // optional
  irt_int ra_inner_max_iter_num;         // optional
  double gr_prefer_wire_unit;            // optional
  double gr_via_unit;                    // optional
  double gr_corner_unit;                 // optional
  double gr_access_history_cost_unit;    // optional
  double gr_resource_history_cost_unit;  // optional
  irt_int gr_max_iter_num;               // optional
  double ta_prefer_wire_unit;            // optional
  double ta_nonprefer_wire_unit;         // optional
  double ta_corner_unit;                 // optional
  double ta_pin_distance_unit;           // optional
  double ta_group_distance_unit;         // optional
  double ta_block_and_pin_unit;          // optional
  double ta_reserved_via_unit;           // optional
  double ta_other_panel_unit;            // optional
  double ta_self_panel_unit;             // optional
  double ta_history_cost_unit;           // optional
  irt_int ta_model_max_iter_num;         // optional
  irt_int ta_panel_max_iter_num;         // optional
  double dr_prefer_wire_unit;            // optional
  double dr_nonprefer_wire_unit;         // optional
  double dr_via_unit;                    // optional
  double dr_corner_unit;                 // optional
  double dr_block_and_pin_unit;          // optional
  double dr_known_panel_unit;            // optional
  double dr_reserved_via_unit;           // optional
  double dr_other_box_unit;              // optional
  double dr_self_box_unit;               // optional
  double dr_history_cost_unit;           // optional
  irt_int dr_model_max_iter_num;         // optional
  irt_int dr_box_max_iter_num;           // optional
  irt_int vr_max_iter_num;               // optional
  /////////////////////////////////////////////
  // **********        RT         ********** //
  std::string log_file_path;         // building
  irt_int bottom_routing_layer_idx;  // building
  irt_int top_routing_layer_idx;     // building
  // **********    DataManager    ********** //
  std::string dm_temp_directory_path;  // building
  // **********  DetailedRouter   ********** //
  std::string dr_temp_directory_path;  // building
  // **********    GDSPlotter     ********** //
  std::string gp_temp_directory_path;  // building
  // **********   GlobalRouter    ********** //
  std::string gr_temp_directory_path;  // building
  // **********   PinAccessor     ********** //
  std::string pa_temp_directory_path;  // building
  // ********   ResourceAllocator   ******** //
  std::string ra_temp_directory_path;  // building
  // **********   TrackAssigner   ********** //
  std::string ta_temp_directory_path;  // building
  // **********  UniversalRouter  ********** //
  std::string ur_temp_directory_path;  // building
  // ********** ViolationRepairer ********** //
  std::string vr_temp_directory_path;  // building
  /////////////////////////////////////////////
};

}  // namespace irt
