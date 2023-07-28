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
#include "DRCChecker.hpp"

#include "DRCRect.hpp"
#include "RTAPI.hpp"
#include "RTUtil.hpp"

namespace irt {

// public

void DRCChecker::initInst()
{
  if (_dc_instance == nullptr) {
    _dc_instance = new DRCChecker();
  }
}

DRCChecker& DRCChecker::getInst()
{
  if (_dc_instance == nullptr) {
    LOG_INST.error(Loc::current(), "The instance not initialized!");
  }
  return *_dc_instance;
}

void DRCChecker::destroyInst()
{
  if (_dc_instance != nullptr) {
    delete _dc_instance;
    _dc_instance = nullptr;
  }
}

// function

std::vector<DRCRect> DRCChecker::getDRCRectList(irt_int net_idx, std::vector<Segment<LayerCoord>>& segment_list)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();

  std::vector<DRCRect> drc_rect_list;
  for (Segment<LayerCoord>& segment : segment_list) {
    LayerCoord& first_coord = segment.get_first();
    LayerCoord& second_coord = segment.get_second();

    irt_int first_layer_idx = first_coord.get_layer_idx();
    irt_int second_layer_idx = second_coord.get_layer_idx();
    if (first_layer_idx != second_layer_idx) {
      RTUtil::sortASC(first_layer_idx, second_layer_idx);
      for (irt_int layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
        ViaMaster& via_master = layer_via_master_list[layer_idx].front();

        LayerRect& above_enclosure = via_master.get_above_enclosure();
        LayerRect offset_above_enclosure(RTUtil::getOffsetRect(above_enclosure, first_coord), above_enclosure.get_layer_idx());
        drc_rect_list.emplace_back(net_idx, offset_above_enclosure, true);

        LayerRect& below_enclosure = via_master.get_below_enclosure();
        LayerRect offset_below_enclosure(RTUtil::getOffsetRect(below_enclosure, first_coord), below_enclosure.get_layer_idx());
        drc_rect_list.emplace_back(net_idx, offset_below_enclosure, true);

        for (PlanarRect& cut_shape : via_master.get_cut_shape_list()) {
          LayerRect offset_cut_shape(RTUtil::getOffsetRect(cut_shape, first_coord), via_master.get_cut_layer_idx());
          drc_rect_list.emplace_back(net_idx, offset_cut_shape, false);
        }
      }
    } else {
      irt_int half_width = routing_layer_list[first_layer_idx].get_min_width() / 2;
      LayerRect wire_rect(RTUtil::getEnlargedRect(first_coord, second_coord, half_width), first_layer_idx);
      drc_rect_list.emplace_back(net_idx, wire_rect, true);
    }
  }
  return drc_rect_list;
}

std::vector<DRCRect> DRCChecker::getDRCRectList(irt_int net_idx, MTree<PHYNode>& phy_node_tree)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();

  std::vector<DRCRect> drc_rect_list;
  for (TNode<PHYNode>* phy_node_node : RTUtil::getNodeList(phy_node_tree)) {
    PHYNode& phy_node = phy_node_node->value();
    if (phy_node.isType<WireNode>()) {
      WireNode& wire_node = phy_node.getNode<WireNode>();
      PlanarRect wire_rect = RTUtil::getEnlargedRect(wire_node.get_first(), wire_node.get_second(), wire_node.get_wire_width() / 2);
      drc_rect_list.emplace_back(net_idx, LayerRect(wire_rect, wire_node.get_layer_idx()), true);
    } else if (phy_node.isType<ViaNode>()) {
      ViaNode& via_node = phy_node.getNode<ViaNode>();
      ViaMasterIdx& via_master_idx = via_node.get_via_master_idx();
      ViaMaster& via_master = layer_via_master_list[via_master_idx.get_below_layer_idx()][via_master_idx.get_via_idx()];

      LayerRect& above_enclosure = via_master.get_above_enclosure();
      LayerRect offset_above_enclosure(RTUtil::getOffsetRect(above_enclosure, via_node), above_enclosure.get_layer_idx());
      drc_rect_list.emplace_back(net_idx, offset_above_enclosure, true);

      LayerRect& below_enclosure = via_master.get_below_enclosure();
      LayerRect offset_below_enclosure(RTUtil::getOffsetRect(below_enclosure, via_node), below_enclosure.get_layer_idx());
      drc_rect_list.emplace_back(net_idx, offset_below_enclosure, true);

      for (PlanarRect& cut_shape : via_master.get_cut_shape_list()) {
        LayerRect offset_cut_shape(RTUtil::getOffsetRect(cut_shape, via_node), via_master.get_cut_layer_idx());
        drc_rect_list.emplace_back(net_idx, offset_cut_shape, false);
      }
    }
  }
  return drc_rect_list;
}

void* DRCChecker::initRegionQuery()
{
}

void DRCChecker::addEnvRectList(void* region_query, const DRCRect& drc_rect)
{
}

void DRCChecker::addEnvRectList(void* region_query, const std::vector<DRCRect>& drc_rect_list)
{
}

void DRCChecker::delEnvRectList(void* region_query, const DRCRect& drc_rect)
{
}

void DRCChecker::delEnvRectList(void* region_query, const std::vector<DRCRect>& drc_rect_list)
{
}

bool DRCChecker::hasViolation(void* region_query, const DRCRect& drc_rect)
{
}

bool DRCChecker::hasViolation(void* region_query, const std::vector<DRCRect>& drc_rect_list)
{
}

std::map<std::string, int> DRCChecker::getViolation(void* region_query)
{
}

std::map<std::string, int> DRCChecker::getViolation(void* region_query, const std::vector<DRCRect>& drc_rect_list)
{
}

std::vector<LayerRect> DRCChecker::getMaxScope(const std::vector<DRCRect>& drc_rect_list)
{
}

std::vector<LayerRect> DRCChecker::getMinScope(const std::vector<DRCRect>& drc_rect_list)
{
}

std::vector<LayerRect> DRCChecker::getMaxScope(const DRCRect& drc_rect)
{
}

std::vector<LayerRect> DRCChecker::getMinScope(const DRCRect& drc_rect)
{
}

void DRCChecker::plotRegionQuery(void* region_query, const std::vector<DRCRect>& drc_rect_list)
{
}

// private

DRCChecker* DRCChecker::_dc_instance = nullptr;

}  // namespace irt
