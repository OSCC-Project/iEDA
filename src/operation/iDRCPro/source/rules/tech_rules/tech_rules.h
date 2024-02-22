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
#pragma once

#include "idm.h"
#include "tech_rule_layer.h"

#define DrcTechRuleInst idrc::TechRules::getInst()

namespace idrc {
class TechRules
{
 public:
  static TechRules* getInst()
  {
    if (_instance == nullptr) {
      _instance = new TechRules();
    }
    return _instance;
  }

  static void destroyInst();
  void set_inited() { _b_inited = true; }
  bool hasInited() { return _b_inited; }

  std::map<idb::IdbLayer*, ConditionRuleLayer*>& get_rule_routing_layers_map() { return _rule_routing_layers; }
  std::map<idb::IdbLayer*, ConditionRuleLayer*>& get_rule_cut_layers_map() { return _rule_cut_layers; }

  ConditionRuleLayer* get_rule_routing_layer(idb::IdbLayer* layer) { return _rule_routing_layers[layer]; }
  ConditionRuleLayer* get_rule_cut_layer(idb::IdbLayer* layer) { return _rule_cut_layers[layer]; }

  int getMinArea(idb::IdbLayer* layer);
  std::vector<std::shared_ptr<idb::routinglayer::Lef58Area>>& getLef58AreaList(idb::IdbLayer* layer);
  int getMinEnclosedArea(idb::IdbLayer* layer);

  int getMinSpacing(idb::IdbLayer* layer, int width = -1);

  ///

 private:
  static TechRules* _instance;
  bool _b_inited = false;

  std::map<idb::IdbLayer*, ConditionRuleLayer*>
      _rule_routing_layers;  /// int : routing layer id, ConditionRuleLayer : all rule map for one routing layer
  std::map<idb::IdbLayer*, ConditionRuleLayer*>
      _rule_cut_layers;  /// int : cut layer id, ConditionRuleLayer : all rule map for one cut layer

  TechRules() {}
  ~TechRules() = default;
};

}  // namespace idrc