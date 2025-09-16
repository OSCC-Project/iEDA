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
#include "pdn_via.h"

#include "idm.h"

namespace ipdn {

int32_t PdnVia::transUnitDB(double value)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = idb_design->get_layout();

  return idb_layout != nullptr ? idb_layout->transUnitDB(value) : -1;
}

/**
 * @brief
 * 在连接处于不同层的电源线时，需要使用到通孔，该函数用于在生成via类型的电源线时提供via
 *
 * @param layer_cut
 * @param width_design
 * @param height_design
 * @param direction
 * @return IdbVia*
 */
idb::IdbVia* PdnVia::findVia(idb::IdbLayerCut* layer_cut, int32_t width_design, int32_t height_design, idb::IdbLayerDirection direction)
{
  auto idb_design = dmInst->get_idb_design();
  auto via_list = idb_design->get_via_list();

  auto direction_str = direction == IdbLayerDirection::kNone ? "" : (direction == IdbLayerDirection::kHorizontal ? "_h" : "_v");

  std::string via_name = layer_cut->get_name() + "_" + std::to_string(width_design) + "x" + std::to_string(height_design) + direction_str;

  idb::IdbVia* via_find = via_list->find_via(via_name);
  if (via_find == nullptr) {
    via_find = createVia(layer_cut, width_design, height_design, via_name, direction);
  }
  return via_find;
}

/**
 * @brief 如果在工艺文件中没有找到合适规格的via，会根据上下层的线宽生成新的通孔
 *
 * @param layer_cut
 * @param width_design
 * @param height_design
 * @param direction
 * @param via_name
 * @return IdbVia*
 */
idb::IdbVia* PdnVia::createVia(idb::IdbLayerCut* layer_cut, int32_t width_design, int32_t height_design, std::string via_name,
                               idb::IdbLayerDirection direction)
{
  auto idb_design = dmInst->get_idb_design();
  auto via_list = idb_design->get_via_list();

  via_name = layer_cut->get_name() + "_" + std::to_string(width_design) + "x" + std::to_string(height_design);

  return via_list->createVia(via_name, layer_cut, width_design, height_design, direction);
}

/**
 * @brief 创建电源线段所用的通孔
 *
 * @param layer
 * @param route_width
 * @param wire_shape_type
 * @param coord
 * @param via
 * @return IdbSpecialWireSegment*
 */
idb::IdbSpecialWireSegment* PdnVia::createSpecialWireVia(idb::IdbLayer* layer, int32_t route_width, idb::IdbWireShapeType wire_shape_type,
                                                         idb::IdbCoordinate<int32_t>* coord, idb::IdbVia* via)
{
  idb::IdbSpecialWireSegment* segment_via = new idb::IdbSpecialWireSegment();
  segment_via->set_is_via(true);
  segment_via->add_point(coord->get_x(), coord->get_y());
  segment_via->set_layer(layer);
  segment_via->set_shape_type(idb::IdbWireShapeType::kStripe);
  segment_via->set_layer_as_new();
  segment_via->set_route_width(0);
  idb::IdbVia* via_new = segment_via->copy_via(via);
  if (via_new != nullptr) {
    via_new->set_coordinate(coord);
  }

  segment_via->set_bounding_box();
  return segment_via;
}

/**
 * @Brief : add via by cut layer name
 * @param  net_name
 * @param  cut_layer_name
 * @param  coord_x
 * @param  coord_y
 * @param  width
 * @param  height
 * @return true
 * @return false
 */
bool PdnVia::addSegmentVia(std::string net_name, std::string cut_layer_name, int32_t coord_x, int32_t coord_y, int32_t width,
                           int32_t height)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = dmInst->get_idb_layout();
  auto idb_layer_list = idb_layout->get_layers();
  auto idb_pdn_list = idb_design->get_special_net_list();

  idb::IdbSpecialNet* net = idb_pdn_list->find_net(net_name);
  if (net == nullptr) {
    std::cout << "Error : can't find the net. " << std::endl;
    return false;
  }

  idb::IdbSpecialWire* wire
      = net->get_wire_list()->get_num() > 0 ? net->get_wire_list()->find_wire(0) : net->get_wire_list()->add_wire(nullptr);
  if (wire == nullptr) {
    std::cout << "Error : can't get the wire." << std::endl;
    return false;
  }

  auto cut_layer = dynamic_cast<idb::IdbLayerCut*>(idb_layer_list->find_layer(cut_layer_name));

  auto via_find = findVia(cut_layer, width, height);
  if (via_find == nullptr) {
    std::cout << "Error : can not find VIA matchs." << std::endl;
    return false;
  }

  idb::IdbLayer* layer_top = via_find->get_top_layer_shape().get_layer();
  idb::IdbCoordinate<int32_t>* middle = new idb::IdbCoordinate<int32_t>(coord_x, coord_y);
  idb::IdbSpecialWireSegment* segment_via = createSpecialWireVia(layer_top, 0, idb::IdbWireShapeType::kStripe, middle, via_find);
  wire->add_segment(segment_via);

  return segment_via == nullptr ? false : true;
}
/**
 * @Brief : add via by metal layer names
 * @param  net_name
 * @param  top_metal
 * @param  bottom_metal
 * @param  coord_x
 * @param  coord_y
 * @param  width
 * @param  height
 * @return true
 * @return false
 */
bool PdnVia::addSegmentVia(std::string net_name, std::string top_metal, std::string bottom_metal, int32_t coord_x, int32_t coord_y,
                           int32_t width, int32_t height)
{
  auto idb_layout = dmInst->get_idb_layout();
  auto idb_layer_list = idb_layout->get_layers();

  /// find all the cut layer betweem top_metal and bottom_metal
  std::vector<idb::IdbLayerCut*> cut_layer_list = idb_layer_list->find_cut_layer_list(top_metal, bottom_metal);
  if (cut_layer_list.size() <= 0) {
    return false;
  }

  /// add via to each cut layer
  for (auto layer : cut_layer_list) {
    if (layer->is_cut()) {
      if (!addSegmentVia(net_name, layer->get_name(), coord_x, coord_y, width, height)) {
        std::cout << "[Pdn Error] : failed to add via for " << layer->get_name() << std::endl;
        return false;
      }
    }
  }

  return true;
}

bool PdnVia::addSegmentVia(std::string net_name, std::string top_metal, std::string bottom_metal, double coord_x, double coord_y,
                           int32_t width, int32_t height)
{
  int dbu_x = transUnitDB(coord_x);
  int dbu_y = transUnitDB(coord_y);

  return addSegmentVia(net_name, top_metal, bottom_metal, dbu_x, dbu_y, width, height);
}

}  // namespace ipdn