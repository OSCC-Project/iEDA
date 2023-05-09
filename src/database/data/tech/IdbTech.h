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
#ifndef IDB_TECH
#define IDB_TECH

#include <iostream>
#include <memory>
#include <vector>

#include "IdbTechLayer.h"
#include "IdbTechVia.h"
#include "IdbTechViaRule.h"

namespace idb {
  class IdbTech {
   public:
    IdbTech();
    ~IdbTech();

    // getter
    IdbTechRoutingLayerList *get_tech_routing_layer_list() { return _tech_routing_layer_list.get(); }
    IdbTechCutLayerList *get_cut_layer_list() { return _tech_cut_layer_list.get(); }
    // setter
    void addTechRoutingLayer(std::unique_ptr<IdbTechRoutingLayer> &layer) {
      _tech_routing_layer_list->addTechRoutingLayer(layer);
    }
    void addTechCutLayer(std::unique_ptr<IdbTechCutLayer> &layer) { _tech_cut_layer_list->addTechCutLayer(layer); }
    void addViaRule(std::unique_ptr<IdbTechViaRule> &viaRule) { _tech_via_rule_list->addViaRule(viaRule); }
    void addVia(std::unique_ptr<IdbTechVia> &via) { _tech_via_list->addVia(via); }

    IdbTechRoutingLayer *getRoutingLayer(const std::string name);
    IdbTechRoutingLayer *getRoutingLayer(int layerId);

    IdbTechCutLayer *getCutLayer(const std::string name);
    IdbTechCutLayer *getCutLayer(int layerId);
    void initLayerId();

    void print();
    void printVia();
    void printViaRule();

    int getRoutingLayerSpacing(int layerId);
    int getCutLayerSpacing(int bottomLayerId);

    IdbTechVia *getOneCutVia(int cutLayerId, int originX, int originY);
    std::vector<IdbTechVia *> getOneCutViaList(int cutLayerId, int originX, int originY);
    std::vector<IdbTechVia *> getTwoCutViaList(int cutLayerId, int originX, int originY);
    IdbTechVia generateArrayCutVia(int cutLayerId, int row, int column);

   private:
    // std::vector<std::unique_ptr<IdbTechLayer>> _tech_layers;
    std::unique_ptr<IdbTechRoutingLayerList> _tech_routing_layer_list;
    std::unique_ptr<IdbTechCutLayerList> _tech_cut_layer_list;
    std::unique_ptr<IdbTechViaRuleList> _tech_via_rule_list;
    std::unique_ptr<IdbTechViaList> _tech_via_list;
  };

}  // namespace idb

#endif
