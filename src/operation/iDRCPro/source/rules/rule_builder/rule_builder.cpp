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

#include "rule_builder.h"

#include "rule_condition_area.h"
#include "rule_condition_edge.h"
#include "rule_condition_spacing.h"
#include "rule_condition_width.h"
#include "tech_rules.h"

namespace idrc {

/// build tech rule at the begining
void DrcRuleBuilder::build()
{
  auto* inst = DrcTechRuleInst->getInst();

  /// init condition map
  if (false == inst->hasInited()) {
    initRoutingLayerRules();
    initCutLayerRules();
    inst->set_inited();
  }
}

void DrcRuleBuilder::initRoutingLayerRules()
{
  /// build rule map
  auto& rule_routing_layers = DrcTechRuleInst->get_rule_routing_layers_map();
  IdbBuilder* builder = dmInst->get_idb_builder();
  idb::IdbLayout* layout = builder->get_lef_service()->get_layout();
  auto& idb_layers = layout->get_layers()->get_routing_layers();
  for (auto* idb_layer : idb_layers) {
    idb::IdbLayerRouting* idb_routing_layer = dynamic_cast<idb::IdbLayerRouting*>(idb_layer);

    /// create rule layer
    ConditionRuleLayer* rule_layer = new ConditionRuleLayer();
    rule_routing_layers[idb_routing_layer->get_name()] = rule_layer;

    buildRoutingLayerArea(rule_layer, idb_routing_layer);
    buildRoutingLayerSpacing(rule_layer, idb_routing_layer);
    buildRoutingLayerWidth(rule_layer, idb_routing_layer);
    buildRoutingLayerEdge(rule_layer, idb_routing_layer);
  }
}

/**
 * build area condition rule map
 */
void DrcRuleBuilder::buildRoutingLayerArea(ConditionRuleLayer* rule_layer, idb::IdbLayerRouting* idb_routing_layer)
{
  /// default min area
  auto build_min_area = [](idb::IdbLayerRouting* idb_routing_layer, RulesMapArea* rule_map) {
    int except_edge_length = idb_routing_layer->get_area() / idb_routing_layer->get_min_width();
    ConditionRuleArea* rule_min_area = new ConditionRuleArea(RuleType::kArea, except_edge_length);
    rule_map->set_condition_rule(RuleType::kAreaMin, except_edge_length, static_cast<ConditionRule*>(rule_min_area));
    rule_map->set_default_rule(except_edge_length, rule_min_area);
  };

  /// lef58 rule, optional
  auto build_min_area_lef58 = [](idb::IdbLayerRouting* idb_routing_layer, RulesMapArea* rule_map) {
    auto& area_rule_lef58 = idb_routing_layer->get_lef58_area();
    for (auto& rule_lef58 : area_rule_lef58) {
      int except_edge_length = rule_lef58.get()->get_except_edge_length().get()->get_max_edge_length();
      ConditionRuleAreaLef58* condition_rule = new ConditionRuleAreaLef58(RuleType::kAreaLef58, except_edge_length, rule_lef58.get());

      rule_map->set_condition_rule(RuleType::kAreaLef58, except_edge_length, static_cast<ConditionRule*>(condition_rule));
    }
  };

  auto* rule_map = new RulesMapArea(RuleType::kArea);

  /// default min area
  build_min_area(idb_routing_layer, rule_map);

  /// lef58 rule, optional
  build_min_area_lef58(idb_routing_layer, rule_map);

  /// set rule map to layer
  rule_layer->set_condition(RuleType::kArea, static_cast<RulesConditionMap*>(rule_map));
}

void DrcRuleBuilder::buildRoutingLayerSpacing(ConditionRuleLayer* rule_layer, idb::IdbLayerRouting* idb_routing_layer)
{
  /// lef5.8 P459
  /// spacing range
  /// key words : SPACING, RANGE
  auto build_spacing_range = [](idb::IdbLayerRouting* idb_routing_layer, RulesMapSpacing* rule_map) {
    auto* idb_spacing_list = idb_routing_layer->get_spacing_list();
    if (idb_spacing_list == nullptr) {
      return;
    }

    for (idb::IdbLayerSpacing* idb_spacing : idb_spacing_list->get_spacing_list()) {
      ConditionRuleSpacingRange* spacing_range
          = new ConditionRuleSpacingRange(RuleType::kSpacingRange, idb_spacing->get_min_spacing(), idb_spacing);

      rule_map->set_condition_rule(RuleType::kSpacingRange, idb_spacing->get_min_spacing(), static_cast<ConditionRule*>(spacing_range));
    }
  };

  /// build rule map
  auto* rule_map = new RulesMapSpacing(RuleType::kSpacing);

  /// spacing range
  build_spacing_range(idb_routing_layer, rule_map);

  /// set rule map to layer
  rule_layer->set_condition(RuleType::kSpacing, static_cast<RulesConditionMap*>(rule_map));
}

/**
 * build width condition rule map
 */
void DrcRuleBuilder::buildRoutingLayerWidth(ConditionRuleLayer* rule_layer, idb::IdbLayerRouting* idb_routing_layer)
{
  /// default width
  auto build_routing_layer_width = [](idb::IdbLayerRouting* idb_routing_layer, RulesMapWidth* rule_map) {
    ConditionRuleWidth* rule_width_min = new ConditionRuleWidth(RuleType::kWidth, idb_routing_layer->get_min_width());
    rule_map->set_condition_rule(RuleType::kWidth, idb_routing_layer->get_min_width(), static_cast<ConditionRule*>(rule_width_min));

    ConditionRuleWidth* rule_width_max = new ConditionRuleWidth(RuleType::kWidth, idb_routing_layer->get_max_width());
    rule_map->set_condition_rule(RuleType::kWidth, idb_routing_layer->get_max_width(), static_cast<ConditionRule*>(rule_width_max));

    ConditionRuleWidth* rule_width_default = new ConditionRuleWidth(RuleType::kWidth, idb_routing_layer->get_width());
    rule_map->set_default_rule(idb_routing_layer->get_width(), rule_width_default);
  };

  /// lef5.8 P474
  /// parallel run length
  /// keywords : SPACINGTABLE, PARALLELRUNLENGTH
  auto build_spacing_parallel_run_length = [](idb::IdbLayerRouting* idb_routing_layer, RulesMapWidth* rule_map) {
    auto idb_table = idb_routing_layer->get_spacing_table();
    if (idb_table == nullptr) {
      return;
    }

    if (idb_table->is_parallel()) {
      /// get prl table
      auto idb_table_prl = idb_table->get_parallel();

      auto& idb_width_list = idb_table_prl->get_width_list();
      auto& prl_length_list = idb_table_prl->get_parallel_length_list();
      auto& idb_spacing_array = idb_table_prl->get_spacing_table();
      for (size_t i = 0; i < idb_spacing_array.size(); ++i) {
        /// get width
        int width = idb_width_list[i];

        ConditionRuleSpacingPRL* spacing_prl = nullptr;
        spacing_prl = new ConditionRuleSpacingPRL(RuleType::kWidthPRLTable, width);

        for (size_t j = 0; j < idb_spacing_array[i].size(); ++j) {
          /// get prl
          int prl_length = prl_length_list[j];

          /// get spacing
          int spacing = idb_spacing_array[i][j];

          spacing_prl->set_spacing(prl_length, spacing);
        }

        rule_map->set_condition_rule(RuleType::kWidthPRLTable, width, static_cast<ConditionRule*>(spacing_prl));
      }
    }
  };

  /// jog to jog
  auto build_spacing_jogtojog = [](idb::IdbLayerRouting* idb_routing_layer, RulesMapWidth* rule_map) {
    auto idb_rule_jog = idb_routing_layer->get_lef58_spacingtable_jogtojog();
    if (idb_rule_jog == nullptr) {
      return;
    }

    for (auto& row : idb_rule_jog->get_width_list()) {
      int width = row.get_width();

      ConditionRuleJogToJog* rule_jog = new ConditionRuleJogToJog(RuleType::kWidthJogToJog, width, idb_rule_jog.get());

      for (auto& idb_width : idb_rule_jog->get_width_list()) {
        rule_jog->addWidth(idb_width.get_width(), &idb_width);
      }

      rule_map->set_condition_rule(RuleType::kWidthJogToJog, width, static_cast<ConditionRule*>(rule_jog));
    }
  };

  auto* rule_map = new RulesMapWidth(RuleType::kWidth);

  /// default width
  build_routing_layer_width(idb_routing_layer, rule_map);

  /// spacing prl
  build_spacing_parallel_run_length(idb_routing_layer, rule_map);

  /// jog to jog
  build_spacing_jogtojog(idb_routing_layer, rule_map);

  /// set rule map to layer
  rule_layer->set_condition(RuleType::kWidth, static_cast<RulesConditionMap*>(rule_map));
}

/**
 * build edge condition rule map
 */
void DrcRuleBuilder::buildRoutingLayerEdge(ConditionRuleLayer* rule_layer, idb::IdbLayerRouting* idb_routing_layer)
{
  /// default min step
  auto build_min_step = [](idb::IdbLayerRouting* idb_routing_layer, RulesMapEdge* rule_map) {
    auto idb_min_step = idb_routing_layer->get_min_step();
    if (idb_min_step == nullptr) {
      return;
    }

    ConditionRuleMinStep* rule_min_step
        = new ConditionRuleMinStep(RuleType::kEdgeMinStep, idb_min_step->get_min_step_length(), idb_min_step.get());
    rule_map->set_condition_rule(RuleType::kEdgeMinStep, idb_min_step->get_min_step_length(), static_cast<ConditionRule*>(rule_min_step));
  };

  /// lef58 min step
  auto build_min_step_lef58 = [](idb::IdbLayerRouting* idb_routing_layer, RulesMapEdge* rule_map) {
    auto& idb_min_step_lef58_list = idb_routing_layer->get_lef58_min_step();
    for (auto& idb_min_step_lef58 : idb_min_step_lef58_list) {
      ConditionRuleMinStepLef58* rule_min_step_lef58
          = new ConditionRuleMinStepLef58(RuleType::kEdgeMinStepLef58, idb_min_step_lef58->get_min_step_length(), idb_min_step_lef58.get());
      rule_map->set_condition_rule(RuleType::kEdgeMinStepLef58, idb_min_step_lef58->get_min_step_length(),
                                   static_cast<ConditionRule*>(rule_min_step_lef58));
    }
  };

  /// notch
  auto build_spacing_notch = [](idb::IdbLayerRouting* idb_routing_layer, RulesMapEdge* rule_map) {
    auto idb_rule_notch = idb_routing_layer->get_lef58_spacing_notchlength();
    if (idb_rule_notch == nullptr) {
      return;
    }

    ConditionRuleNotch* rule_notch = new ConditionRuleNotch(RuleType::kEdgeNotch, idb_rule_notch->get_min_spacing(), idb_rule_notch.get());
    rule_map->set_condition_rule(RuleType::kEdgeNotch, idb_rule_notch->get_min_spacing(), static_cast<ConditionRule*>(rule_notch));
  };

  /// end of line
  auto build_spacing_eol = [](idb::IdbLayerRouting* idb_routing_layer, RulesMapEdge* rule_map) {
    auto idb_rule_eol_list = idb_routing_layer->get_lef58_spacing_eol_list();
    for (auto idb_rule_eol : idb_rule_eol_list) {
      ConditionRuleEOL* rule_eol = new ConditionRuleEOL(RuleType::kEdgeEOL, idb_rule_eol->get_eol_width(), idb_rule_eol.get());

      rule_map->set_condition_rule(RuleType::kEdgeEOL, idb_rule_eol->get_eol_width(), static_cast<ConditionRule*>(rule_eol));
    }
  };

  auto* rule_map = new RulesMapEdge(RuleType::kEdge);

  /// default min step
  build_min_step(idb_routing_layer, rule_map);

  /// lef58 min step
  build_min_step_lef58(idb_routing_layer, rule_map);

  /// notch
  build_spacing_notch(idb_routing_layer, rule_map);

  /// end of line
  build_spacing_eol(idb_routing_layer, rule_map);

  /// set rule map to layer
  rule_layer->set_condition(RuleType::kEdge, static_cast<RulesConditionMap*>(rule_map));
}

//TODO: cut layer rules need tobe added
void DrcRuleBuilder::initCutLayerRules()
{
}

}  // namespace idrc