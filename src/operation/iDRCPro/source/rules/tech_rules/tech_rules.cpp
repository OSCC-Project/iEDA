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

#include "tech_rules.h"

namespace idrc {

TechRules* TechRules::_instance = nullptr;

void TechRules::destroyInst()
{
  if (_instance != nullptr) {
    delete _instance;
    _instance = nullptr;
  }
}

idb::IdbLayerRouting* TechRules::getRoutingLayer(int layer_index)
{
  // TODO: pointer is null
  IdbBuilder* builder = dmInst->get_idb_builder();
  idb::IdbLayout* layout = builder->get_lef_service()->get_layout();
  idb::IdbLayers* idb_layers = layout->get_layers();
  idb::IdbLayer* idb_layer = idb_layers->find_routing_layer(layer_index);  // ! layer index equals to layer order???

  idb::IdbLayerRouting* idb_routing_layer = dynamic_cast<idb::IdbLayerRouting*>(idb_layer);
  return idb_routing_layer;
}

int TechRules::getMinArea(int layer_index)
{
  idb::IdbLayerRouting* idb_routing_layer = getRoutingLayer(layer_index);

  return idb_routing_layer->get_area();
}

std::vector<std::shared_ptr<idb::routinglayer::Lef58Area>>& TechRules::getLef58AreaList(int layer_index)
{
  idb::IdbLayerRouting* idb_routing_layer = getRoutingLayer(layer_index);

  return idb_routing_layer->get_lef58_area();
}

int TechRules::getMinEnclosedArea(int layer_index)
{
  idb::IdbLayerRouting* idb_routing_layer = getRoutingLayer(layer_index);

  vector<IdbMinEncloseArea>& min_area_list = idb_routing_layer->get_min_enclose_area_list()->get_min_area_list();

  return min_area_list.size() > 0 ? min_area_list[0]._area : 0;
}

int TechRules::getMinSpacing(int layer_index, int width)
{
  idb::IdbLayerRouting* idb_routing_layer = getRoutingLayer(layer_index);

  return idb_routing_layer->get_spacing(width);
}

}  // namespace idrc