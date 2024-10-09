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
#include "ieco_data.h"

#include "IdbLayer.h"
#include "IdbVias.h"

namespace ieco {

EcoData::EcoData()
{
}

EcoData::~EcoData()
{
  _via_layers.clear();
}

EcoDataViaLayer& EcoData::get_via_layer(const std::string layer)
{
  if (_via_layers.empty() || _via_layers.find(layer) == _via_layers.end()) {
    _via_layers.insert(std::make_pair(layer, EcoDataViaLayer()));
  }
  return _via_layers[layer];
}

EcoDataViaLayer& EcoData::get_via_layer(idb::IdbVia* idb_via)
{
  std::string cut_layer = idb_via->get_cut_layer_shape().get_layer()->get_name();
  if (_via_layers.empty() || _via_layers.find(cut_layer) == _via_layers.end()) {
    _via_layers.insert(std::make_pair(cut_layer, EcoDataViaLayer()));

    idb::IdbLayerRouting* bottom_layer = dynamic_cast<idb::IdbLayerRouting*>(idb_via->get_bottom_layer_shape().get_layer());
    auto direction_bottom
        = bottom_layer->get_direction() == idb::IdbLayerDirection::kHorizontal ? Direction::kHorizontal : Direction::kVertical;
    _via_layers[cut_layer].set_prefer_direction_bottom(direction_bottom);

    idb::IdbLayerRouting* top_layer = dynamic_cast<idb::IdbLayerRouting*>(idb_via->get_top_layer_shape().get_layer());
    auto direction_top = top_layer->get_direction() == idb::IdbLayerDirection::kHorizontal ? Direction::kHorizontal : Direction::kVertical;
    _via_layers[cut_layer].set_prefer_direction_top(direction_top);
  }
  return _via_layers[cut_layer];
}

}  // namespace ieco