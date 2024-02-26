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

#include "rule_builder.h"

namespace idrc {

TechRules* TechRules::_instance = nullptr;

void TechRules::init()
{
  DrcRuleBuilder builder;
  builder.build();
}

void TechRules::destroyInst()
{
  if (_instance != nullptr) {
    delete _instance;
    _instance = nullptr;
  }
}

int TechRules::getMinArea(idb::IdbLayer* layer)
{
  idb::IdbLayerRouting* idb_routing_layer = dynamic_cast<idb::IdbLayerRouting*>(layer);

  return idb_routing_layer->get_area();
}

std::vector<std::shared_ptr<idb::routinglayer::Lef58Area>>& TechRules::getLef58AreaList(idb::IdbLayer* layer)
{
  idb::IdbLayerRouting* idb_routing_layer = dynamic_cast<idb::IdbLayerRouting*>(layer);

  return idb_routing_layer->get_lef58_area();
}

int TechRules::getMinEnclosedArea(idb::IdbLayer* layer)
{
  idb::IdbLayerRouting* idb_routing_layer = dynamic_cast<idb::IdbLayerRouting*>(layer);

  vector<IdbMinEncloseArea>& min_area_list = idb_routing_layer->get_min_enclose_area_list()->get_min_area_list();

  return min_area_list.size() > 0 ? min_area_list[0]._area : 0;
}

int TechRules::getMinSpacing(idb::IdbLayer* layer, int width)
{
  idb::IdbLayerRouting* idb_routing_layer = dynamic_cast<idb::IdbLayerRouting*>(layer);

  return idb_routing_layer->get_spacing(width);
}

}  // namespace idrc