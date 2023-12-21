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

#include "IdbLayer.h"
#include "condition.h"
#include "condition_connectivity.h"
#include "idrc_data.h"
#include "rule_enum.h"

namespace idrc {
/**
 * check overlap for each layout(same layer) between different sub layout(different net)
 */
bool DrcRuleConditionConnectivity::checkOverlap()
{
  bool b_result = true;

  /// check routing layer
  auto& engine_layouts = get_engine()->get_engine_manager()->get_engine_layouts(LayoutType::kRouting);
  // layer id : indicate layer
  // engine_layout : all nets shapes in the indicate layer
  for (auto& [layer, engine_layout] : engine_layouts) {
    /// sub_layouts : indicate shapes for all nets in one layer
    auto& sub_layouts = engine_layout->get_sub_layouts();
    /// no overlap in this layer
    if ((int) sub_layouts.size() < MAX_CMP_NUM) {
      continue;
    }

    /// compare polygons between different sub layout
    /// iter_1 iter_2 : indicate shapes for one net in the same layer
    for (auto iter_1 = sub_layouts.begin(); iter_1 != sub_layouts.end(); iter_1++) {
      auto iter_2 = iter_1;
      iter_2++;
      for (; iter_2 != sub_layouts.end(); iter_2++) {
        auto* engine_1 = iter_1->second->get_engine();
        auto* engine_2 = iter_2->second->get_engine();
        bool b_result_sub = engine_1->checkOverlap(engine_2);
        if (b_result_sub == false) {
          std::cout << "Check overlap layer_id = " << layer->get_id() << " net id 1= " << iter_1->first << " net id 2= " << iter_2->first
                    << std::endl;
        }

        b_result &= b_result_sub;
      }
    }
  }
  return b_result;
}

}  // namespace idrc