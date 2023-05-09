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
#include "Tech.h"
namespace idrc {
// /**
//  * @brief find cut layer by cut layer index
//  *
//  * @param layer_id
//  * @return DrcCutLayer*
//  */
// DrcCutLayer* Tech::getCutLayerById(int layer_id)
// {
//   for (auto& layer : _drc_cut_layer_list) {
//     if (layer_id == layer->get_layer_id()) {
//       return layer;
//     }
//   }
//   return nullptr;
// }
/**
 * @brief get cut layer required spacing by index
 *
 * @param cutLayerId
 * @return int
 */
int Tech::getCutSpacing(int cutLayerId)
{
  for (auto& cut_layer : _drc_cut_layer_list) {
    if (cut_layer->get_layer_id() == cutLayerId) {
      return cut_layer->get_cut_spacing();
    }
  }
  return -1;
}

// DrcRoutingLayer* Tech::add_routing_layer()
// {
//   DrcRoutingLayer* routingLayer = new DrcRoutingLayer();
//   _drc_routing_layer_list.push_back(routingLayer);
//   return routingLayer;
// }
// DrcCutLayer* Tech::add_cut_layer()
// {
//   DrcCutLayer* cutLayer = new DrcCutLayer();
//   _drc_cut_layer_list.push_back(cutLayer);
//   return cutLayer;
// }

// DrcVia* Tech::add_via()
// {
//   DrcVia* via = new DrcVia();
//   _via_lib.push_back(via);
//   return via;
// }
int Tech::getRoutingWidth(int routingLayerId)
{
  for (auto& routing_layer : _drc_routing_layer_list) {
    if (routing_layer->get_layer_id() == routingLayerId) {
      return routing_layer->get_default_width();
    }
  }
  std::cout << "[Tech Error!] in getRoutingWidth:This layer does not exist" << std::endl;
  return -1;
}
int Tech::getRoutingSpacing(int routingLayerId, int width)
{
  for (auto& routing_layer : _drc_routing_layer_list) {
    if (routing_layer->get_layer_id() == routingLayerId) {
      return routing_layer->getRoutingSpacing(width);
    }
  }
  std::cout << "[Tech Error!] in getRoutingSpacing:This layer does not exist" << std::endl;
  return -1;
}

int Tech::getRoutingMaxRequireSpacing(int routingLayerId, DrcRect* target_rect)
{
  for (auto& routing_layer : _drc_routing_layer_list) {
    if (routing_layer->get_layer_id() == routingLayerId) {
      return routing_layer->getLayerMaxRequireSpacing(target_rect);
    }
  }
  std::cout << "[Tech Error!] in getRoutingMaxRequireSpacing:This layer does not exist" << std::endl;
  return -1;
}

int Tech::getRoutingMinWidth(int routingLayerId)
{
  for (auto& routing_layer : _drc_routing_layer_list) {
    if (routing_layer->get_layer_id() == routingLayerId) {
      return routing_layer->get_min_width();
    }
  }
  std::cout << "[Tech Error!] in getRoutingMaxRequireSpacing:This layer does not exist" << std::endl;
  return -1;
}

// int Tech::getRoutingMinArea(int routingLayerId)
// {
//   for (auto& routing_layer : _drc_routing_layer_list) {
//     if (routing_layer->get_layer_id() == routingLayerId) {
//       return routing_layer->get_min_area();
//     }
//   }
//   std::cout << "[Tech Error!] in getRoutingMaxRequireSpacing:This layer does not exist" << std::endl;
//   return -1;
// }

int Tech::getRoutingMinEnclosedArea(int routingLayerId)
{
  for (auto& routing_layer : _drc_routing_layer_list) {
    if (routing_layer->get_layer_id() == routingLayerId) {
      return routing_layer->get_min_enclosed_area();
    }
  }
  std::cout << "[Tech Error!] in getRoutingMaxRequireSpacing:This layer does not exist" << std::endl;
  return -1;
}
// LayerDirection Tech::getLayerDirection(int routingLayerId)
// {
//   for (auto& routing_layer : _drc_routing_layer_list) {
//     if (routing_layer->get_layer_id() == routingLayerId) {
//       return routing_layer->get_direction();
//     }
//   }
//   std::cout << "[Tech Error!] in getRoutingMaxRequireSpacing:This layer does not exist" << std::endl;
//   return LayerDirection::kNone;
// }

////clear
void Tech::clear_drc_routing_layer_list()
{
  for (DrcRoutingLayer* drc_routing_layer : _drc_routing_layer_list) {
    if (drc_routing_layer != nullptr) {
      delete drc_routing_layer;
      drc_routing_layer = nullptr;
    }
  }
}

void Tech::clear_drc_cut_layer_list()
{
  for (DrcCutLayer* drc_cut_layer : _drc_cut_layer_list) {
    if (drc_cut_layer != nullptr) {
      delete drc_cut_layer;
      drc_cut_layer = nullptr;
    }
  }
}
void Tech::clear_via_lib()
{
  for (DrcVia* via : _via_lib) {
    if (via != nullptr) {
      delete via;
      via = nullptr;
    }
  }
}
///////////////////////
// int Tech::getRoutingLayerIdByLayerName(const std::string& name)
// {
//   for (auto& routing_layer : _drc_routing_layer_list) {
//     if (routing_layer->get_name() == name) {
//       return routing_layer->get_layer_id();
//     }
//   }
//   std::cout << "[Tech Error!] in getLayerIdByLayerName:This layer does not exist" << std::endl;
//   return -1;
// }

std::string Tech::getCutLayerNameById(int layer_id)
{
  for (auto& cut_layer : _drc_cut_layer_list) {
    if (cut_layer->get_layer_id() == layer_id) {
      return cut_layer->get_name();
    }
  }
  return "";
}

std::string Tech::getRoutingLayerNameById(int layer_id)
{
  // return _drc_cut_layer_list[layer_id]->get_name();

  for (auto& routing_layer : _drc_routing_layer_list) {
    if (routing_layer->get_layer_id() == layer_id) {
      return routing_layer->get_name();
    }
  }
  return "";
}

int Tech::getLayerIdByLayerName(const std::string& name)
{
  for (auto& cut_layer : _drc_cut_layer_list) {
    if (cut_layer->get_name() == name) {
      return cut_layer->get_layer_id();
    }
  }
  for (auto& routing_layer : _drc_routing_layer_list) {
    if (routing_layer->get_name() == name) {
      return routing_layer->get_layer_id();
    }
  }
  return -1;
}

std::pair<bool, int> Tech::getLayerInfoByLayerName(const std::string& name)
{
  for (auto& routing_layer : _drc_routing_layer_list) {
    if (routing_layer->get_name() == name) {
      return std::make_pair(true, routing_layer->get_layer_id());
    }
  }
  for (auto& cut_layer : _drc_cut_layer_list) {
    if (cut_layer->get_name() == name) {
      return std::make_pair(false, cut_layer->get_layer_id());
    }
  }
  std::cout << "[Tech Error!] in getLayerIdByLayerName:This layer does not exist" << std::endl;
  return std::make_pair(false, -1);
}

}  // namespace idrc