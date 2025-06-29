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
 * @author Jianrong Su
 * @brief Implementation of TemplateLib class
 * @version 1.0
 * @date 2025-06-23
 */

#include "TemplateLib.hh"
#include "log/Log.hh"

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
  _horizontal_templates.push_back(gen_single_template(StripeDirection::kHorizontal, 8000.0, 1600.0, 19200.0, 8000.0));
  _horizontal_templates.push_back(gen_single_template(StripeDirection::kHorizontal, 8000.0, 1600.0, 38400.0, 8000.0));
  _horizontal_templates.push_back(gen_single_template(StripeDirection::kHorizontal, 8000.0, 1600.0, 38400.0, 27200.0));

  // Vertical direction templates
  _vertical_templates.push_back(gen_single_template(StripeDirection::kVertical, 8000.0, 1600.0, 19200.0, 8000.0));
  _vertical_templates.push_back(gen_single_template(StripeDirection::kVertical, 8000.0, 1600.0, 38400.0, 8000.0));
  _vertical_templates.push_back(gen_single_template(StripeDirection::kVertical, 8000.0, 1600.0, 38400.0, 27200.0));
}

void TemplateLib::gen_template_libs_from_config(const PNPConfig* config)
{
  if (!config) {
    LOG_WARNING << "PNPConfig is null, using default hardcoded templates." << std::endl;
    gen_template_libs();
    return;
  }

  // Clear existing templates
  _horizontal_templates.clear();
  _vertical_templates.clear();

  // Get templates from config
  const auto& horizontal_templates = config->get_horizontal_templates();
  const auto& vertical_templates = config->get_vertical_templates();

  // Check if both template vectors are empty
  if (horizontal_templates.empty() && vertical_templates.empty()) {
    LOG_WARNING << "No templates found in configuration, using default hardcoded templates." << std::endl;
    gen_template_libs();
    return;
  }

  // Process horizontal templates
  for (const auto& template_config : horizontal_templates) {
    _horizontal_templates.push_back(
      gen_single_template(
        StripeDirection::kHorizontal,
        template_config.width,
        template_config.pg_offset,
        template_config.space,
        template_config.offset
      )
    );
  }

  // Process vertical templates
  for (const auto& template_config : vertical_templates) {
    _vertical_templates.push_back(
      gen_single_template(
        StripeDirection::kVertical,
        template_config.width,
        template_config.pg_offset,
        template_config.space,
        template_config.offset
      )
    );
  }

  LOG_INFO << "Generated " << _horizontal_templates.size() << " horizontal templates and "
           << _vertical_templates.size() << " vertical templates from configuration." << std::endl;
}

}  // namespace ipnp 