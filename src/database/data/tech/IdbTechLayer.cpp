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
#include "IdbTechLayer.h"

namespace idb {

  /**
   * @brief Construct a new Idb Tech Layer:: Idb Tech Layer object
   *
   */
  IdbTechLayer::IdbTechLayer() : _type(LayerTypeEnum::kUNKNOWN), _layer_id(0) { }
  IdbTechLayer::~IdbTechLayer() { }

  /**
   * @brief Construct a new Idb Tech Routing Layer:: Idb Tech Routing Layer object
   *
   */
  IdbTechRoutingLayer::IdbTechRoutingLayer()
      : IdbTechLayer(),
        _direction(LayerDirEnum::kUNKNOWN),
        _pitch(-1),
        _offset(-1),
        _thickness(-1),
        _resistance(-1),
        _capacitance(-1),
        _edge_capcitance(-1),
        _width(-1),
        _max_width_check(nullptr),
        _min_width_check(nullptr),
        _min_area_check(nullptr),
        _density_check(nullptr),
        _min_step_check(nullptr),
        _spacing_table_prl_Check(nullptr),
        _spacing_table_two_widtth_check(nullptr),
        _spacing_check(nullptr) {
    _spacing_range_check_list   = std::make_unique<IdbSpacingRangeCheckList>();
    _spacing_eol_check_list     = std::make_unique<IdbSpacingEolCheckList>();
    _spacing_samenet_check_list = std::make_unique<IdbSpacingSamenetCheckList>();
  }

  IdbTechRoutingLayer::~IdbTechRoutingLayer() { }

  void IdbTechRoutingLayer::set_direction(IdbLayerDirection dir) {
    if (dir == IdbLayerDirection::kHorizontal) {
      _direction = LayerDirEnum::kHORITIONAL;
    } else if (dir == IdbLayerDirection::kVertical) {
      _direction = LayerDirEnum::kVERTICAL;
    } else {
      _direction = LayerDirEnum::kUNKNOWN;
    }
  }
  // interface

  std::vector<IdbSpacingRangeCheck *> IdbTechRoutingLayer::getSpacingRangeCheckList() {
    std::vector<IdbSpacingRangeCheck *> vecCheck = _spacing_range_check_list->getIdbSpacingRangeChecks();
    return vecCheck;
  }
  int IdbTechRoutingLayer::getRoutingSpacing(int width) {
    std::vector<IdbSpacingRangeCheck *> rangeCheckList = getSpacingRangeCheckList();
    for (auto rangeCheck : rangeCheckList) {
      // std::cout << "spacing::" << rangeCheck->get_min_spacing() << " width1::" << rangeCheck->get_min_width()
      //           << " width2::" << rangeCheck->get_max_width() << std::endl;
      if (width >= rangeCheck->get_min_width() && width <= rangeCheck->get_max_width()) {
        return rangeCheck->get_min_spacing();
      }
    }
    return getRoutingSpacing();
  }
  /**
   * @brief Construct a new Idb Tech Cut Layer:: Idb Tech Cut Layer object
   *
   */
  IdbTechCutLayer::IdbTechCutLayer() : IdbTechLayer(), _width(-1), _array_spacing_check(nullptr), _enclosure_check(nullptr) {
    _dccurrent_density_check_list = std::make_unique<IdbDccurrentDensityCheckList>();
  }
  IdbTechCutLayer::~IdbTechCutLayer() { }

  /********************************************************/
  /********************************************************/

  IdbTechRoutingLayer *IdbTechRoutingLayerList::findRoutingLayer(const std::string &name) {
    for (auto &layer : _tech_routing_layers) {
      if (name == layer->get_name()) {
        return layer.get();
      }
    }
    return nullptr;
  }
  IdbTechRoutingLayer *IdbTechRoutingLayerList::findRoutingLayer(int layerId) {
    for (auto &layer : _tech_routing_layers) {
      if (layerId == layer->get_layer_id()) {
        return layer.get();
      }
    }
    return nullptr;
  }
  void IdbTechRoutingLayerList::initRoutingLayerId() {
    for (size_t i = 0; i < _tech_routing_layers.size(); ++i) {
      _tech_routing_layers[i]->set_layer_id(i + 1);
    }
  }

  void IdbTechRoutingLayerList::printRoutingLayer() {
    for (auto &layer : _tech_routing_layers) {
      std::cout << "layer name ::" << layer->get_name() << " layerId ::" << layer->get_layer_id() << std::endl;
    }
  }
  /********************************************************/
  /********************************************************/
  IdbTechCutLayer *IdbTechCutLayerList::findCutLayer(const std::string &name) {
    for (auto &layer : _tech_cut_layers) {
      if (name == layer->get_name()) {
        return layer.get();
      }
    }
    return nullptr;
  }
  IdbTechCutLayer *IdbTechCutLayerList::findCutLayer(int layerId) {
    for (auto &layer : _tech_cut_layers) {
      if (layerId == layer->get_layer_id()) {
        return layer.get();
      }
    }
    return nullptr;
  }
  void IdbTechCutLayerList::initCutLayerId() {
    for (size_t i = 0; i < _tech_cut_layers.size(); ++i) {
      _tech_cut_layers[i]->set_layer_id(i);
    }
  }

  void IdbTechCutLayerList::printCutLayer() {
    for (auto &layer : _tech_cut_layers) {
      std::cout << "layer name ::" << layer->get_name() << " layerId ::" << layer->get_layer_id() << std::endl;
    }
  }

}  // namespace idb
