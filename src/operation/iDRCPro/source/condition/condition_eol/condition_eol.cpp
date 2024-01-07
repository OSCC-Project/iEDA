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

#include "condition_eol.h"

#include "condition.h"
#include "idrc_violation.h"
#include "idrc_violation_enum.h"
#include "idrc_violation_manager.h"
#include "rule_condition_edge.h"
#include "rule_enum.h"
#include "tech_rules.h"

namespace idrc {

bool DrcRuleConditionEOL::checkFastMode()
{
  bool b_result = true;

  b_result &= checkSpacingEOL();

  return b_result;
}

bool DrcRuleConditionEOL::checkCompleteMode()
{
  bool b_result = true;

  b_result &= checkSpacingEOL();

  return b_result;
}

bool DrcRuleConditionEOL::checkSpacingEOL()
{
  bool b_result = true;

  auto& check_map = _condition_manager->get_check_map(RuleType::kEdgeEOL);
  for (auto& [layer, check_list] : check_map) {
    // get rule step map
    auto* rule_routing_layer = DrcTechRuleInst->get_rule_routing_layer(layer);
    if (rule_routing_layer == nullptr) {
      continue;
    }
    auto* rule_map = rule_routing_layer->get_condition_map(RuleType::kEdge);
    auto& rule_eol_map = rule_map->get_rule_map(RuleType::kEdgeEOL);

    // handle eol edges
    for (auto& point_pair : check_list->get_points()) {
      // match rule EOL spacing
      if (!checkSpacingEOL(point_pair.first, point_pair.second, layer, rule_eol_map)) {
        b_result = false;
      }
    }
  }

  return b_result;
}

bool DrcRuleConditionEOL::checkSpacingEOL(DrcBasicPoint* point_left, DrcBasicPoint* point_right, idb::IdbLayer* layer,
                                          std::map<int, idrc::ConditionRule*> rule_eol_map)
{
  bool b_result = true;

  bool is_vertical = point_left->get_x() == point_right->get_x();

  // violation rect data
  int llx = std::min(point_left->get_x(), point_right->get_x());
  int lly = std::min(point_left->get_y(), point_right->get_y());
  int urx = std::max(point_left->get_x(), point_right->get_x());
  int ury = std::max(point_left->get_y(), point_right->get_y());
  std::set<int> net_ids;
  net_ids.insert(point_left->get_id());
  net_ids.insert(point_right->get_id());

  int step_edge_length = point_left->distance(point_right);
  // find rule and check
  for (auto& [value, rule_eol] : rule_eol_map) {
    if (value <= step_edge_length) {
      continue;
    }

    // get rule data
    auto* condition_rule_eol = static_cast<ConditionRuleEOL*>(rule_eol);
    // todo
    int a = 0;
  }

  return b_result;
}

}  // namespace idrc