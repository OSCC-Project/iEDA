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

#include "rule_condition_width.h"

namespace idrc {
/// @brief  check width & prl rules contains condition
/// @param width
/// @param prl_length
/// @return true : match condition == has violation; false == no violation
// bool ConditionRuleSpacingPRL::isMatchCondition(int width, int prl_length)
// {
//   return width > _width && prl_length > _prl_length;
// }
/// @brief check jot to jog rules contains condition
/// @param value
/// @return
std::vector<idb::routinglayer::Lef58SpacingTableJogToJog::Width*> ConditionRuleJogToJog::findWidthRule(int wire_max_width)
{
  std::vector<idb::routinglayer::Lef58SpacingTableJogToJog::Width*> width_list;
  for (auto& width_class : _width_map) {
    /// find the 1st larger witdh in the list
    if (wire_max_width >= width_class.first) {
      width_list.emplace_back(width_class.second);
    }
  }

  return width_list;
}

}  // namespace idrc