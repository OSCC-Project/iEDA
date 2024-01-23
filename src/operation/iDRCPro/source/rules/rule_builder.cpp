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
  }
}

void DrcRuleBuilder::buildJogConditions(idb::IdbLayer* layer, idb::IdbLayerRouting* idb_routing_layer)
{
  auto idb_rule_jog = idb_routing_layer->get_lef58_spacingtable_jogtojog();
  if (idb_rule_jog == nullptr) {
    return;
  }

  // find max within
  int within = 0;
  for (auto& row : idb_rule_jog->get_width_list()) {
    int current_within = row.get_par_within();
    if (current_within > within) {
      within = current_within;
    }
  }

  // create condition
  uint64_t trigger_sequence = ConditionSequence::kESE | ConditionSequence::kESW_WSE;
  ConditionSequenceJog* sequence = new ConditionSequenceJog(within, trigger_sequence, ConditionSequence::kSEW_WES,
                                                            ConditionSequence::kESE | ConditionSequence::kESW_WSE);
  ConditionDetailJog* detail = new ConditionDetailJog(idb_rule_jog.get());

  Condition* condition = new Condition(sequence, detail);

  DrcTechRuleInst->get_condition_routing_layers(layer)[trigger_sequence].emplace_back(condition);
}

void DrcRuleBuilder::initCutLayerRules()
{
}

}  // namespace idrc