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
#include "EnclosedAreaCheck.h"

#include "DRCUtil.h"
#include "DrcConfig.h"
#include "DrcDesign.h"
#include "RegionQuery.h"
#include "Tech.h"

namespace idrc {

/**
 * @brief 检查目标线网书否存在孔洞面积违规
 *
 * @param target_net
 */
void EnclosedAreaCheck::checkEnclosedArea(DrcNet* target_net)
{
  _layer_to_polygons_map.clear();
  initLayerPolygonSet(target_net);
  checkEnclosedArea();
}

/**
 * @brief 将线网每一层的矩形包括Pin，Via，Segment矩形合并为导体多边形
 *
 * @param target_net
 */
void EnclosedAreaCheck::initLayerPolygonSet(DrcNet* target_net)
{
  initLayerToPolygonSetFromRoutingRects(target_net);
  initLayerToPolygonSetFromPinRects(target_net);
}

/**
 * @brief 将目标检查线网中绕线矩形包括Via，segment合成多边形
 *
 * @param target_net
 */
void EnclosedAreaCheck::initLayerToPolygonSetFromRoutingRects(DrcNet* target_net)
{
  for (auto& [layerId, routing_rect_list] : target_net->get_layer_to_routing_rects_map()) {
    for (auto routing_rect : routing_rect_list) {
      BoostRect routingRect = DRCUtil::getBoostRect(routing_rect);
      _layer_to_polygons_map[layerId] += routingRect;
    }
  }
}

/**
 * @brief 将目标检查线网中Pin矩形参与到多边形的合成过程中
 *
 * @param target_net
 */
void EnclosedAreaCheck::initLayerToPolygonSetFromPinRects(DrcNet* target_net)
{
  for (auto& [layerId, pin_rect_list] : target_net->get_layer_to_pin_rects_map()) {
    for (auto pin_rect : pin_rect_list) {
      BoostRect pinRect = DRCUtil::getBoostRect(pin_rect);
      _layer_to_polygons_map[layerId] += pinRect;
    }
  }
}

/**
 * @brief 检查线网中各个孔洞的面积是否满足EnclosedArea，可以通过Boost相关数据与接口获得孔洞
 *
 * @param hole_polygon_list
 * @param requre_enclosed_area
 * @param layerId
 */
void EnclosedAreaCheck::checkEnclosedAreaFromHolePlygonList(const std::vector<PolygonWithHoles>& hole_polygon_list,
                                                            int requre_enclosed_area, int layerId)
{
  //遍历线网中的每一个带孔洞多边形
  for (auto& hole_polygon : hole_polygon_list) {
    //遍历线网中带孔洞多边形中孔洞进行检查
    for (auto hole_it = hole_polygon.begin_holes(); hole_it != hole_polygon.end_holes(); ++hole_it) {
      auto hole_polygon = *hole_it;
      int hole_area = bp::area(hole_polygon);
      if (hole_area < requre_enclosed_area) {
        _region_query->addViolation(ViolationType::kEnclosedArea);
        // BoostRect bounding_box;
        // bp::extents(bounding_box, hole_polygon);
        // Rectangle violation_box = DRCUtil::getRectangleFromBoostRect(bounding_box);
        // add_spot(layerId, violation_box, ViolationType::kEnclosedArea);
      }

      // IDB还未解析enclosed area的值！！！！目前只要检测到有环路就是有孔洞就加入违规存储，如果iDB解析了enclosed
      // area的值去掉下面代码，打开上面注释代码即可
      //目前DR绕线生成的孔洞都是小孔洞，基本上都是违规的
      // BoostRect bounding_box;
      // bp::extents(bounding_box, hole_polygon);
      // DrcRectangle violation_box = DRCUtil::getRectangleFromBoostRect(bounding_box);
      // add_spot(layerId, violation_box, ViolationType::kEnclosedArea);
    }
  }
}

/**
 * @brief 检查目标线网下各个孔洞的面积是否符合孔洞面积要求
 * 通过Boost接口将一般的多边形数据转化为带孔洞的多边形数据，然后遍历每个多边形中的孔洞进行EnclosedArea检查，Boost中提供了获取每个带孔洞多边形中孔洞的接口
 *
 */
void EnclosedAreaCheck::checkEnclosedArea()
{
  std::vector<PolygonWithHoles> hole_polygon_list;
  for (auto& [layerId, polygon_set] : _layer_to_polygons_map) {
    if (_tech->getRoutingMinEnclosedArea(layerId) == 0) {
      continue;
    }
    int requre_enclosed_area = _tech->getRoutingMinEnclosedArea(layerId);

    hole_polygon_list.clear();
    //将合并的Boost多边形转数据化为Boost孔洞多边形数据
    polygon_set.get(hole_polygon_list);
    //对带孔洞的多边形数据进行最小包围面积检查
    checkEnclosedAreaFromHolePlygonList(hole_polygon_list, requre_enclosed_area, layerId);
  }
}

/**
 * @brief 将不满足EnclosedArea需求的孔洞外接矩形作为违规标记矩形进行存储
 *
 * @param layerId 违规发生的金属层
 * @param vialation_box 不满足EnclosedArea需求的孔洞外接矩形
 * @param type 违规类型
 */
void EnclosedAreaCheck::add_spot(int layerId, const DrcRectangle<int>& vialation_box, ViolationType type)
{
  DrcSpot spot;

  DrcRect* drc_rect = new DrcRect();
  drc_rect->set_owner_type(RectOwnerType::kSpotMark);
  drc_rect->set_layer_id(layerId);
  drc_rect->set_rectangle(vialation_box);

  spot.set_violation_type(ViolationType::kEnclosedArea);
  spot.add_spot_rect(drc_rect);
  _routing_layer_to_spots_map[layerId].emplace_back(spot);
}

/**
 * @brief 孔洞面积检查模块初始化
 *
 * @param config
 * @param tech
 */
void EnclosedAreaCheck::init(DrcConfig* config, Tech* tech)
{
  _config = config;
  _tech = tech;
}

/**
 * @brief 孔洞面积检查模块重置
 *
 */
void EnclosedAreaCheck::reset()
{
  _layer_to_polygons_map.clear();

  for (auto& [LayerId, spot_list] : _routing_layer_to_spots_map) {
    for (auto& spot : spot_list) {
      spot.clearSpotRects();
    }
  }
  _routing_layer_to_spots_map.clear();
}

/**
 * @brief 获得孔洞面积违规的数目
 *
 * @return int 孔洞面积违规的数目
 */

int EnclosedAreaCheck::get_enclosed_area_violation_num()
{
  int count = 0;
  for (auto& [layerId, spot_list] : _routing_layer_to_spots_map) {
    count += spot_list.size();
  }
  return count;
}

////**********************interact with iRT********************////
void EnclosedAreaCheck::checkEnclosedArea(const LayerNameToRTreeMap& layer_to_rects_tree_map)
{
  initLayerPolygonSet(layer_to_rects_tree_map);
  checkEnclosedArea();
}

void EnclosedAreaCheck::initLayerPolygonSet(const LayerNameToRTreeMap& layer_to_rects_tree_map)
{
  for (auto& [layerName, rtree] : layer_to_rects_tree_map) {
    int layerId = _tech->getLayerIdByLayerName(layerName);
    for (auto it = rtree.begin(); it != rtree.end(); ++it) {
      DrcRect* target_rect = it->second;
      _layer_to_polygons_map[layerId] += DRCUtil::getBoostRect(target_rect);
    }
  }
}

}  // namespace idrc