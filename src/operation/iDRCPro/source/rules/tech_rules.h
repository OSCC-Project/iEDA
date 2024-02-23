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

#include <stdint.h>

#include <optional>

#include "IdbLayer.h"
#include "condition.h"
#include "idm.h"

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

  enum TechType : uint64_t
  {
    kNone = 0,
    kIntersection = 1,
    kSelf = 2
  };

  static void destroyInst();
  void set_inited() { _b_inited = true; }
  bool hasInited() { return _b_inited; }

  void init();

  // getter
  std::map<int, std::vector<Condition*>>& get_condition_routing_layers(idb::IdbLayer* layer)
  {
    return _condition_routing_layers[layer->get_id()];
  }

  std::optional<std::vector<Condition*>*> get_condition_with_width(idb::IdbLayer* layer, int width)
  {
    for (auto& width_condition_map : _condition_routing_layers[layer->get_id()]) {
      if (width_condition_map.first < width) {
        return &width_condition_map.second;
      }
    }
    return std::nullopt;
  }

 private:
  static TechRules* _instance;
  bool _b_inited = false;

  std::map<int, std::map<int, std::vector<Condition*>>> _condition_routing_layers;  // int: layer id, int: width

  TechRules() {}
  ~TechRules();
};

}  // namespace idrc