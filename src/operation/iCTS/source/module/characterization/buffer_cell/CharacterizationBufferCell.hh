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
/**
 * @file CharacterizationBufferCell.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Buffer cell electrical limits and pin roles used by CTS segment characterization.
 */

#pragma once
// IWYU pragma: private, include "characterization/buffer_cell/CharacterizationBufferCell.hh"

#include <string>

namespace icts {

struct CharacterizationBufferCell
{
  std::string cell_master;
  double max_cap_pf = 0.0;  // Drive-strength proxy used for ordering and explicit auto-derivation.
  double input_cap_pf = 0.0;
  double input_slew_limit_ns = 0.0;
  double input_slew_table_axis_max_ns = 0.0;
  double output_cap_limit_pf = 0.0;
  double output_cap_table_axis_max_pf = 0.0;
  double cell_height_um = 0.0;
  std::string input_pin;
  std::string output_pin;

  auto operator==(const CharacterizationBufferCell& rhs) const -> bool = default;
};

}  // namespace icts
