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

#include "RTHeader.hpp"

namespace irt {

class Config
{
 public:
  Config() = default;
  ~Config() = default;
  /////////////////////////////////////////////
  // **********        RT         ********** //
  std::string temp_directory_path;   // required
  int32_t thread_number;             // optional
  std::string bottom_routing_layer;  // optional
  std::string top_routing_layer;     // optional
  int32_t output_inter_result;       // optional
  int32_t enable_notification;       // optional
  int32_t enable_timing;             // optional
  /////////////////////////////////////////////
  // **********        RT         ********** //
  std::string log_file_path;         // building
  int32_t bottom_routing_layer_idx;  // building
  int32_t top_routing_layer_idx;     // building
  // **********    DataManager    ********** //
  std::string dm_temp_directory_path;  // building
  // **********     DRCEngine     ********** //
  std::string de_temp_directory_path;  // building
  // **********     GDSPlotter    ********** //
  std::string gp_temp_directory_path;  // building
  // **********    PinAccessor    ********** //
  std::string pa_temp_directory_path;  // building
  // ********     SupplyAnalyzer    ******** //
  std::string sa_temp_directory_path;  // building
  // ********   TopologyGenerator   ******** //
  std::string tg_temp_directory_path;  // building
  // **********   LayerAssigner   ********** //
  std::string la_temp_directory_path;  // building
  // **********    SpaceRouter    ********** //
  std::string sr_temp_directory_path;  // building
  // **********   TrackAssigner   ********** //
  std::string ta_temp_directory_path;  // building
  // **********   DetailedRouter  ********** //
  std::string dr_temp_directory_path;  // building
  // ********** ViolationReporter ********** //
  std::string vr_temp_directory_path;  // building
  // **********    EarlyRouter    ********** //
  std::string er_temp_directory_path;  // building
  /////////////////////////////////////////////
};

}  // namespace irt
