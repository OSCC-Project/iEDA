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
 * @file PwrCell.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The power cell, mapping to design inst, used for leakage power
 * analysis.
 * @version 0.1
 * @date 2023-01-18
 */
#pragma once

#include "include/PwrConfig.hh"
#include "netlist/Instance.hh"

namespace ipower {

/**
 * @brief The power cell mapped to netlist instance.
 *
 */
class PwrCell {
 public:
  explicit PwrCell(ista::Instance* design_inst) : _design_inst(design_inst) {}
  ~PwrCell() = default;

  auto* get_design_inst() { return _design_inst; }

 private:
  ista::Instance* _design_inst;
};

}  // namespace ipower