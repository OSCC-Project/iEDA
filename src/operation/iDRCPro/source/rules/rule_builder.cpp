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

#include "condition.h"
#include "condition_detail_jog.h"
#include "condition_sequence_jog.h"
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
  IdbBuilder* builder = dmInst->get_idb_builder();
  idb::IdbLayout* layout = builder->get_lef_service()->get_layout();
  auto& idb_layers = layout->get_layers()->get_routing_layers();
  for (auto* idb_layer : idb_layers) {
    idb::IdbLayerRouting* idb_routing_layer = dynamic_cast<idb::IdbLayerRouting*>(idb_layer);

    buildJogConditions(idb_layer, idb_routing_layer);
    buildSpacingTableConditions(idb_layer, idb_routing_layer);
  }
}

void DrcRuleBuilder::buildJogConditions(idb::IdbLayer* layer, idb::IdbLayerRouting* idb_routing_layer)
{
  auto idb_rule_jog = idb_routing_layer->get_lef58_spacingtable_jogtojog();
  if (idb_rule_jog == nullptr) {
    return;
  }

  // create condition for each width
  for (auto& row : idb_rule_jog->get_width_list()) {
    int width = row.get_width();

    // create condition
    // std::vector<ConditionSequence::SequenceType> trigger_sequence_list{ConditionSequence::kSE_MS_W, ConditionSequence::kSE_MS_SE,
    //                                                                    ConditionSequence::kSE_MS_TE};
    // uint64_t trigger_sequence = 0;
    // for (auto& sequence : trigger_sequence_list) {
    //   trigger_sequence |= sequence;
    // }
    // ConditionSequenceJog* sequence
    //     = new ConditionSequenceJog(width, trigger_sequence, ConditionSequence::kTE_MS_W | ConditionSequence::kTE_MS_TE,
    //                                ConditionSequence::kEE_MS_W | ConditionSequence::kEE_MS_EE | ConditionSequence::kEE_MS_TE);
    // ConditionDetailJog* detail = new ConditionDetailJog(idb_rule_jog.get());

    // Condition* condition = new Condition(sequence, detail);

    // for (auto& sequence : trigger_sequence_list) {
    //   DrcTechRuleInst->get_condition_trigger(layer, sequence).emplace_back(condition);
    // }

    Condition* condition = new Condition(width);
    DrcTechRuleInst->get_condition_routing_layers(layer)[width].emplace_back(condition);
  }
}

void DrcRuleBuilder::buildSpacingTableConditions(idb::IdbLayer* layer, idb::IdbLayerRouting* idb_routing_layer)
{
  auto idb_rule_spacing_table = idb_routing_layer->get_spacing_table();
  if (idb_rule_spacing_table == nullptr) {
    return;
  }

  // find max spacing
  int max_spacing = 0;
  if (idb_rule_spacing_table->is_parallel()) {
    /// get prl table
    auto idb_table_prl = idb_rule_spacing_table->get_parallel();

    auto& idb_width_list = idb_table_prl->get_width_list();
    auto& prl_length_list = idb_table_prl->get_parallel_length_list();
    auto& idb_spacing_array = idb_table_prl->get_spacing_table();
    for (size_t i = 0; i < idb_spacing_array.size(); ++i) {
      for (size_t j = 0; j < idb_spacing_array[i].size(); ++j) {
        /// get spacing
        int spacing = idb_spacing_array[i][j];

        if (spacing > max_spacing) {
          max_spacing = spacing;
        }
      }
    }

    // // create condition
    // std::vector<ConditionSequence::SequenceType> trigger_sequence_list{
    //     ConditionSequence::kSE_MS_W, ConditionSequence::kSE_MS_SE, ConditionSequence::kSE_MS_TE};
    // uint64_t trigger_sequence = 0;
    // for (auto& sequence : trigger_sequence_list) {
    //   trigger_sequence |= sequence;
    // }
    // ConditionSequenceJog* sequence
    //     = new ConditionSequenceJog(within, trigger_sequence, ConditionSequence::kTE_MS_W | ConditionSequence::kTE_MS_TE,
    //                                ConditionSequence::kEE_MS_W | ConditionSequence::kEE_MS_EE | ConditionSequence::kEE_MS_TE);
    // ConditionDetailJog* detail = new ConditionDetailJog(idb_rule_spacing_table.get());

    // Condition* condition = new Condition(sequence, detail);

    // for (auto& sequence : trigger_sequence_list) {
    //   DrcTechRuleInst->get_condition_trigger(layer, sequence).emplace_back(condition);
    // }
  }
}

void DrcRuleBuilder::initCutLayerRules()
{
}

}  // namespace idrc