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

#include "condition_matrix.h"

#include "condition.h"
#include "idrc_data.h"

namespace idrc {

/**
 * check fast mode : ignore some condition checking
 * check complete mode : check all condition
 * return : true = no violation, false = has violation
 */

bool DrcRuleConditionMatrix::check(DrcStratagyType type )
{
  bool b_result = true;

  switch (type) {
    case DrcStratagyType::kCheckFast: {
      b_result = checkFastMode();
    } break;
    case DrcStratagyType::kCheckComplete: {
      b_result = checkCompleteMode();
    } break;
    default:
      break;
  }

  return b_result;
}

}  // namespace idrc