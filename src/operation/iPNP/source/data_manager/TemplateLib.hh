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
 * @author Jianrong Su
 * @brief Generate template blocks.
 * @version 1.0
 * @date 2025-06-23
 */

#pragma once

#include <vector>

#include "PNPConfig.hh"
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

  SingleTemplate gen_single_template(StripeDirection direction, double width, double pg_offset, double space, double offset);

  // Generate template libraries using hardcoded values (legacy method)
  void gen_template_libs();

  // Generate template libraries from configuration
  void gen_template_libs_from_config();

  const std::vector<SingleTemplate>& get_horizontal_templates() const { return _horizontal_templates; }
  const std::vector<SingleTemplate>& get_vertical_templates() const { return _vertical_templates; }

 private:
  std::vector<SingleTemplate> _horizontal_templates;  // Template library for horizontal direction
  std::vector<SingleTemplate> _vertical_templates;    // Template library for vertical direction
};

}  // namespace ipnp
