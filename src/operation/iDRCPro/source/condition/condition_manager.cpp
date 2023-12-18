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

#include "condition_manager.h"

#include "condition.h"
#include "condition_area.h"
#include "condition_connectivity.h"
#include "condition_spacing.h"
#include "drc_basic_point.h"
#include "idrc_data.h"
#include "idrc_violation_manager.h"

namespace idrc {
/**
 * build condition for all stratagy and conditon matrix
 * return : true = no violation, false = has violation
 */
bool DrcConditionManager::buildCondition()
{
  bool b_result = true;

  b_result &= buildConditonConnectivity();
  //   b_result &= buildConditonSpacing();
  b_result &= buildConditonArea();

  //   auto& stratagy_type = _stratagy.get_stratagy_type();
  //   switch (stratagy_type) {
  //     case DrcStratagyType::kCheckFast: {
  //       b_result &= buildConditonConnectivity();
  //       //   b_result &= buildConditonSpacing();
  //       b_result &= buildConditonArea();
  //     } break;
  //     case DrcStratagyType::kCheckComplete: {
  //       b_result &= buildConditonConnectivity();
  //       //   b_result &= buildConditonSpacing();
  //       b_result &= buildConditonArea();
  //     } break;
  //     default:
  //       break;
  //   }

  return b_result;
}
/**
 * build condition for connectivity
 */
bool DrcConditionManager::buildConditonConnectivity()
{
  bool b_result = true;

  DrcRuleConditionConnectivity condition_connectivity(_engine);
  b_result = condition_connectivity.check();

  return b_result;
}
/**
 * build condition for area
 */
bool DrcConditionManager::buildConditonArea()
{
  bool b_result = true;

  DrcRuleConditionArea condition_area(_engine);
  b_result = condition_area.check();

  return b_result;
}
/**
 * build condition for spacing
 */
bool DrcConditionManager::buildConditonSpacing()
{
  bool b_result = true;

  DrcRuleConditionSpacing condition_spacing(_engine);
  b_result = condition_spacing.check();

  return b_result;
}

}  // namespace idrc