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

#include <string>

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

  static void init();

  static void destroyInst();
  void set_inited() { _b_inited = true; }
  bool hasInited() { return _b_inited; }

  std::map<std::string, ConditionRuleLayer*>& get_rule_routing_layers_map() { return _rule_routing_layers; }
  std::map<std::string, ConditionRuleLayer*>& get_rule_cut_layers_map() { return _rule_cut_layers; }

  ConditionRuleLayer* get_rule_routing_layer(std::string layer_name) { return _rule_routing_layers[layer_name]; }
  ConditionRuleLayer* get_rule_cut_layer(std::string layer_name) { return _rule_cut_layers[layer_name]; }

  idb::IdbLayer* findLayer(std::string layer_name)
  {
    auto idb_design = dmInst->get_idb_design();
    auto idb_layout = idb_design->get_layout();
    return idb_layout->get_layers()->find_layer(layer_name);
  }

  int getMinArea(std::string layer_name);
  std::vector<std::shared_ptr<idb::routinglayer::Lef58Area>>& getLef58AreaList(std::string layer_name);
  int getMinEnclosedArea(std::string layer_name);

  int getMinSpacing(std::string layer_name, int width = 0);

  std::shared_ptr<idb::routinglayer::Lef58SpacingTableJogToJog> getJogToJog(std::string layer_name);

  std::shared_ptr<idb::IdbLayerSpacingTable> getSpacingTable(std::string layer_name);

  std::vector<std::shared_ptr<idb::routinglayer::Lef58SpacingEol>> getSpacingEolList(std::string layer_name);

  std::shared_ptr<routinglayer::Lef58CornerFillSpacing> getCornerFillSpacing(std::string layer_name);

  ///

 private:
  static TechRules* _instance;
  bool _b_inited = false;

  std::map<std::string, ConditionRuleLayer*>
      _rule_routing_layers;  /// string : routing layer name, ConditionRuleLayer : all rule map for one routing layer
  std::map<std::string, ConditionRuleLayer*>
      _rule_cut_layers;  /// string : cut layer name, ConditionRuleLayer : all rule map for one cut layer

  TechRules() {}
  ~TechRules() = default;
};

}  // namespace idrc