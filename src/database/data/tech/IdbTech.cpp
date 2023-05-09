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
#include "IdbTech.h"

namespace idb {
  IdbTech::IdbTech() {
    _tech_routing_layer_list = std::make_unique<IdbTechRoutingLayerList>();
    _tech_cut_layer_list     = std::make_unique<IdbTechCutLayerList>();
    _tech_via_rule_list      = std::make_unique<IdbTechViaRuleList>();
    _tech_via_list           = std::make_unique<IdbTechViaList>();
  }
  IdbTech::~IdbTech() {
    // for (auto check : _checks) {
    //   if (check) {
    //     delete check;
    //     check = nullptr;
    //   }
    // }
  }

  // initLayerId
  void IdbTech::initLayerId() {
    _tech_routing_layer_list->initRoutingLayerId();
    _tech_cut_layer_list->initCutLayerId();
  }
  // findlayer
  IdbTechRoutingLayer *IdbTech::getRoutingLayer(const std::string name) {
    return _tech_routing_layer_list->findRoutingLayer(name);
  }
  IdbTechRoutingLayer *IdbTech::getRoutingLayer(int layerId) { return _tech_routing_layer_list->findRoutingLayer(layerId); }

  IdbTechCutLayer *IdbTech::getCutLayer(const std::string name) { return _tech_cut_layer_list->findCutLayer(name); }
  IdbTechCutLayer *IdbTech::getCutLayer(int layerId) { return _tech_cut_layer_list->findCutLayer(layerId); }

  int IdbTech::getRoutingLayerSpacing(int layerId) {
    IdbTechRoutingLayer *layer = getRoutingLayer(layerId);
    return layer->getRoutingSpacing();
  }
  int IdbTech::getCutLayerSpacing(int bottomLayerId) {
    IdbTechCutLayer *layer = getCutLayer(bottomLayerId);
    return layer->getCutSpacing();
  }

  void IdbTech::print() {
    _tech_routing_layer_list->printRoutingLayer();
    _tech_cut_layer_list->printCutLayer();
  }
  void IdbTech::printVia() { _tech_via_list->printVia(); }
  void IdbTech::printViaRule() { _tech_via_rule_list->printViaRule(); }

  IdbTechVia *IdbTech::getOneCutVia(int cutLayerId, int originX, int originY) {
    return _tech_via_list->getOneCutVia(cutLayerId, originX, originY);
  }
  std::vector<IdbTechVia *> IdbTech::getOneCutViaList(int cutLayerId, int originX, int originY) {
    return _tech_via_list->getOneCutViaList(cutLayerId, originX, originY);
  }
  std::vector<IdbTechVia *> IdbTech::getTwoCutViaList(int cutLayerId, int originX, int originY) {
    return _tech_via_list->getTwoCutViaList(cutLayerId, originX, originY);
  }

  IdbTechVia IdbTech::generateArrayCutVia(int cutLayerId, int row, int column) {
    IdbTechViaRule *viaRule = _tech_via_rule_list->getTechViaRule(cutLayerId);
    IdbTechVia via;
    int cutSpacingX          = viaRule->get_cut_spacing_x();
    int cutSpacingY          = viaRule->get_cut_spacing_y();
    int cutRectLowerLeftX    = viaRule->getCutLowerLeftX();
    int cutRectLowerLeftY    = viaRule->getCutLowerLeftY();
    int cutRectUpperRightX   = viaRule->getCutUpperRightX();
    int cutRectUpperRightY   = viaRule->getCutUpperRightY();
    int arrayStartLowerLeftX = 0, arrayStartLowerLeftY = 0;
    int arrayEndUpperRightX = 0, arrayEndUpperRightY = 0;
    for (int i = 0; i < row; ++i) {
      int lly = cutRectLowerLeftY + cutSpacingY * i;
      int ury = cutRectUpperRightY + cutSpacingY * i;
      for (int j = 0; j < column; ++j) {
        int llx = cutRectLowerLeftX + cutSpacingX * j;
        int urx = cutRectUpperRightX + cutSpacingX * j;
        if (i == 0 && j == 0) {
          arrayStartLowerLeftX = llx;
          arrayStartLowerLeftY = lly;
        }
        if (i == row - 1 && j == column - 1) {
          arrayEndUpperRightX = urx;
          arrayEndUpperRightY = ury;
        }
        via.addCutRectList(llx, lly, urx, ury);
      }
    }
    via.setBottomRect(
        arrayStartLowerLeftX - viaRule->getBottomEnclosureX(), arrayStartLowerLeftY - viaRule->getBottomEnclosureY(),
        arrayEndUpperRightX + viaRule->getBottomEnclosureX(), arrayEndUpperRightY + viaRule->getBottomEnclosureY());
    via.setTopRect(arrayStartLowerLeftX - viaRule->getTopEnclosureX(), arrayStartLowerLeftY - viaRule->getTopEnclosureY(),
                   arrayEndUpperRightX + viaRule->getTopEnclosureX(), arrayEndUpperRightY + viaRule->getTopEnclosureY());
    return via;
  }
}  // namespace idb
