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
 * @file TemplateLib.cpp
 * @author Xinhao li
 * @brief Implementation of TemplateLib class
 * @version 0.1
 * @date 2024-07-15
 */

#include "TemplateLib.hh"

namespace ipnp {

SingleTemplate TemplateLib::gen_single_template(StripeDirection direction,
                                           double width,
                                           double pg_offset,
                                           double space,
                                           double offset)
{
  SingleTemplate single_template(direction, PowerType::kVSS, width, pg_offset, space, offset);
  return single_template;
}

void TemplateLib::gen_template_libs()
{
  // Horizontal direction templates
  _horizontal_templates.push_back(gen_single_template(StripeDirection::kHorizontal, 8000.0, 1600.0, 19200.0, 8000.0));  // 5组
  _horizontal_templates.push_back(gen_single_template(StripeDirection::kHorizontal, 8000.0, 1600.0, 38400.0, 8000.0));  // 3组
  _horizontal_templates.push_back(gen_single_template(StripeDirection::kHorizontal, 8000.0, 1600.0, 38400.0, 27200.0)); // 2组

  // Vertical direction templates
  _vertical_templates.push_back(gen_single_template(StripeDirection::kVertical, 8000.0, 1600.0, 19200.0, 8000.0));  // 5组
  _vertical_templates.push_back(gen_single_template(StripeDirection::kVertical, 8000.0, 1600.0, 38400.0, 8000.0));  // 3组
  _vertical_templates.push_back(gen_single_template(StripeDirection::kVertical, 8000.0, 1600.0, 38400.0, 27200.0));  // 2组
  // _vertical_templates.push_back(gen_single_template(StripeDirection::kVertical, 900.0, 1600.0, 19200.0, 8000.0));  // M7特殊层
  
}

}  // namespace ipnp 