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
 * @file TemplateLib.hh
 * @author Xinhao li
 * @brief Generate template blocks. Upper interface is GenPdnTemplate.py
 * @version 0.1
 * @date 2024-07-15
 */

#pragma once

#include <vector>
#include "SingleTemplate.hh"

namespace ipnp {

/**
 * @brief Library for PDN templates
 */
class TemplateLib
{
 public:
  TemplateLib() = default;
  ~TemplateLib() = default;

  /**
   * @brief Generate a single template
   * 
   * @param direction Direction of the template stripes
   * @param width Width of the stripes
   * @param pg_offset Offset between power and ground wires
   * @param space Distance between edges of two VDD wires
   * @param offset Offset from bottom (horizontal) or left (vertical)
   * @return SingleTemplate The generated template
   */
  SingleTemplate gen_single_template(StripeDirection direction,
    double width,
    double pg_offset,
    double space,
    double offset);

  void gen_template_libs();

  const std::vector<SingleTemplate>& get_horizontal_templates() const { return _horizontal_templates; }  
  const std::vector<SingleTemplate>& get_vertical_templates() const { return _vertical_templates; }

 private:
  std::vector<SingleTemplate> _horizontal_templates;  // Template library for horizontal direction
  std::vector<SingleTemplate> _vertical_templates;    // Template library for vertical direction
};

}  // namespace ipnp
