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
#include "idrc_data.h"
#include "idrc_engine.h"
#include "idrc_rule_stratagy.h"

namespace idrc {

class DrcBasicPoint;
class DrcViolationManager;
/**
 * rule conditions are concepts built from tech lef drc rules, it contains a condition matrix to guide condition check orders, the rule
 * matrix index indicates the checking order,
 *
 */

class DrcConditionManager
{
  class LayerCheckList
  {
   public:
    LayerCheckList(idb::IdbLayer* layer) : _layer(layer) {}
    ~LayerCheckList() {}

    void addCheckList(DrcBasicPoint* pt1, DrcBasicPoint* pt2) { _points.emplace_back(std::make_pair(pt1, pt2)); }

   private:
    idb::IdbLayer* _layer = nullptr;
    std::vector<std::pair<DrcBasicPoint*, DrcBasicPoint*>> _points;
  };

 public:
  DrcConditionManager(DrcEngine* engine, DrcViolationManager* violation_manager) : _engine(engine), _violation_manager(violation_manager) {}
  ~DrcConditionManager()
  {
    for (auto& [type, layer_map] : _check_maps) {
      for (auto& [layer, check_list] : layer_map) {
        delete check_list;
      }
    }
  }

  bool buildCondition();

  DrcEngine* get_engine() { return _engine; }
  DrcViolationManager* get_violation_manager() { return _violation_manager; }

  std::map<RuleType, std::map<idb::IdbLayer*, LayerCheckList*>>& get_check_maps() { return _check_maps; }
  std::map<idb::IdbLayer*, LayerCheckList*>& get_check_map(RuleType type) { return _check_maps[type]; }
  // LayerCheckList* get_check_list_routing_layer(idb::IdbLayer* layer) { return get_check_list(RuleType::kRouting, layer); }
  // LayerCheckList* get_check_list_cut_layer(idb::IdbLayer* layer) { return get_check_list(RuleType::kCut, layer); }
  LayerCheckList* get_check_list(RuleType type, idb::IdbLayer* layer)
  {
    auto& layer_map = _check_maps[type];
    if (false == layer_map.contains(layer)) {
      layer_map[layer] = new LayerCheckList(layer);
    }

    return layer_map[layer];
  }

 private:
  DrcEngine* _engine = nullptr;
  DrcViolationManager* _violation_manager;

  std::map<RuleType, std::map<idb::IdbLayer*, LayerCheckList*>> _check_maps;

  /// condition builder
  bool buildConditonConnectivity();
  bool buildConditonArea();
  bool buildConditonSpacing();
};

}  // namespace idrc