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

#include "DRCHeader.hpp"

namespace idrc {

class Config
{
 public:
  Config() = default;
  ~Config() = default;
  //////////////////////////////////////////////
  // **********        DRC         ********** //
  std::string temp_directory_path;    // required
  int32_t thread_number;              // optional
  std::string golden_directory_path;  // optional
  int32_t enable_notification;        // optional
  /////////////////////////////////////////////
  // **********        DRC         ********** //
  std::string log_file_path;  // building
  // **********   RuleValidator    ********** //
  std::string rv_temp_directory_path;  // building
  // **********     GDSPlotter     ********** //
  std::string gp_temp_directory_path;  // building
  //////////////////////////////////////////////
};

}  // namespace idrc
