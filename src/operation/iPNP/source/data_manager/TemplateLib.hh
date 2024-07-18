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

#include "GridManager.hh"

namespace ipnp {

class TemplateLib
{
 public:
  TemplateLib() = default;
  ~TemplateLib() = default;

  auto gen_pdn_template(double width, double space, StripeDirection direction)
  {
    /**
     * @todo generate pdn template according to GenPdnTemplate.py
     * @brief _curr_template = xxx
     */
  }

  auto gen_template_libs()
  {
    /**
     * @todo _curr_template --> _template_libs
     */
  }

 private:
  PDNGridTemplate _curr_template;
  std::vector<PDNGridTemplate> _template_libs;
};

}  // namespace ipnp
