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
/**
 * @File Name: dm_transform.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-11-01
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "idm.h"

namespace idm {
/**
 * @brief move the coordinate(coord_x, coord_y) of cell master to instance position, including moving & rotation
 *
 * @param coord_x coordinate x in cell master
 * @param coord_y coordinate y in cell master
 * @param cell_master_name cell master name
 * @param inst_x position x of instance
 * @param inst_y position y of instance
 * @param idb_orient orient of instance
 */
void DataManager::transformCoordinate(int32_t& coord_x, int32_t& coord_y, std::string cell_master_name, int32_t inst_x, int32_t inst_y,
                                      idb::IdbOrient idb_orient)
{
  auto cell_master_list = _layout->get_cell_master_list();
  auto cell_master = cell_master_list->find_cell_master(cell_master_name);
  if (cell_master == nullptr) {
    return;
  }

  /// instance coordinate
  idb::IdbCoordinate<int32_t> coord(inst_x, inst_y);
  idb::IdbOrientTransform db_transform(idb_orient, &coord, cell_master->get_width(), cell_master->get_height());

  int32_t x = coord_x + inst_x;
  int32_t y = coord_y + inst_y;
  IdbCoordinate<int32_t> trans_coord(x, y);
  db_transform.transformCoordinate(&trans_coord);

  coord_x = trans_coord.get_x();
  coord_y = trans_coord.get_y();
}
/**
 * @brief align coordinate to (0,0)
 *
 * @param coord
 * @return true
 * @return false
 */
bool DataManager::alignCoord(IdbCoordinate<int32_t>* coord)
{
  if (coord == nullptr) {
    return false;
  }

  if (!coord->is_init()) {
    return false;
  }

  auto idb_die = _layout->get_die();
  coord->set_xy(coord->get_x() - idb_die->get_llx(), coord->get_y() - idb_die->get_lly());

  return true;
}
/**
 * @brief align rect to (0,0)
 *
 * @param rect
 * @return true
 * @return false
 */
bool DataManager::alignRect(IdbRect* rect)
{
  if (rect == nullptr) {
    return false;
  }

  if (!rect->is_init()) {
    return false;
  }

  auto idb_die = _layout->get_die();
  rect->moveByStep(-idb_die->get_llx(), -idb_die->get_lly());

  return true;
};

bool DataManager::alignLayerShape(IdbLayerShape* layer_shape)
{
  if (layer_shape == nullptr) {
    return false;
  }

  for (auto rect : layer_shape->get_rect_list()) {
    alignRect(rect);
  }

  return true;
}
/**
 * @brief align pin to (0,0)
 *
 * @param idb_pin
 * @return true
 * @return false
 */
bool DataManager::alignPin(IdbPin* idb_pin)
{
  if (idb_pin == nullptr) {
    return false;
  }

  alignCoord(idb_pin->get_average_coordinate());
  alignCoord(idb_pin->get_location());
  alignCoord(idb_pin->get_grid_coordinate());
  alignRect(idb_pin->get_bounding_box());

  for (auto layer_shape : idb_pin->get_port_box_list()) {
    alignLayerShape(layer_shape);
  }
  return true;
}

bool DataManager::alignSignalSegment(IdbRegularWireSegment* idb_segment)
{
  if (idb_segment == nullptr) {
    return false;
  }

  /// via
  for (auto via : idb_segment->get_via_list()) {
    alignVia(via);
  }

  /// delta rect
  alignRect(idb_segment->get_delta_rect());

  /// segment
  for (auto pt : idb_segment->get_point_list()) {
    alignCoord(pt);
  }

  return true;
}

bool DataManager::alignSpecialSegment(IdbSpecialWireSegment* idb_segment)
{
  if (idb_segment == nullptr) {
    return false;
  }

  alignRect(idb_segment->get_bounding_box());

  /// via
  alignVia(idb_segment->get_via());

  /// segment
  for (auto pt : idb_segment->get_point_list()) {
    alignCoord(pt);
  }

  return true;
}

bool DataManager::alignVia(IdbVia* idb_via)
{
  if (idb_via == nullptr) {
    return false;
  }

  alignCoord(idb_via->get_coordinate());
  alignRect(idb_via->get_bounding_box());

  return true;
}
/**
 * @brief check if die left & bottom coordiante is on (0,0)
 *
 * @return true
 * @return false
 */
bool DataManager::isNeedTransformByDie()
{
  /// if original coordinate is on (0,0), ignore the transform
  auto idb_die = _layout->get_die();
  if (idb_die == nullptr) {
    return false;
  }
  /// if not initialize
  if (idb_die->get_points().size() <= 0) {
    return false;
  }

  /// if original location is on (0,0)
  if (idb_die->get_llx() == 0 && idb_die->get_lly() == 0) {
    return false;
  }

  std::cout << "transform by die" << std::endl;
  return true;
}
/**
 * @brief make die location of left bottom align with (0,0)
 *
 * @return true
 * @return false
 */
bool DataManager::transformByDie()
{
  auto idb_die = _layout->get_die();

  /// core
  auto idb_core = _layout->get_core();
  alignRect(idb_core->get_bounding_box());

  // rows
  auto idb_row_list = _layout->get_rows();
  for (auto idb_row : idb_row_list->get_row_list()) {
    alignCoord(idb_row->get_original_coordinate());
    alignRect(idb_row->get_bounding_box());
  }

  /// GCellGrid
  auto idb_gcell_list = _layout->get_gcell_grid_list();
  for (auto idb_gcell : idb_gcell_list->get_gcell_grid_list()) {
    int alias = idb_gcell->get_direction() == IdbTrackDirection::kDirectionX ? idb_die->get_llx() : idb_die->get_lly();
    idb_gcell->set_start(idb_gcell->get_start() - alias);
  }

  /// TrackGrid
  auto idb_track_list = _layout->get_track_grid_list();
  for (auto idb_track_grid : idb_track_list->get_track_grid_list()) {
    auto track = idb_track_grid->get_track();
    int alias = track->get_direction() == IdbTrackDirection::kDirectionX ? idb_die->get_llx() : idb_die->get_lly();
    track->set_start(track->get_start() - alias);
  }

  /// IO PIN
  auto idb_io_pins = _design->get_io_pin_list();
  for (auto idb_pin : idb_io_pins->get_pin_list()) {
    alignPin(idb_pin);
  }

  /// Instance
  auto idb_inst_list = _design->get_instance_list();
  for (auto idb_inst : idb_inst_list->get_instance_list()) {
    alignRect(idb_inst->get_bounding_box());
    alignCoord(idb_inst->get_coordinate());

    /// instance pin
    auto idb_inst_pin_list = idb_inst->get_pin_list();
    for (auto idb_inst_pin : idb_inst_pin_list->get_pin_list()) {
      alignPin(idb_inst_pin);
    }
    /// instance obs
    for (auto idb_inst_obs : idb_inst->get_obs_box_list()) {
      alignLayerShape(idb_inst_obs);
    }
    /// instance halo
    auto idb_inst_halo = idb_inst->get_halo();
    if (idb_inst_halo != nullptr) {
      alignRect(idb_inst_halo->get_bounding_box());
    }
  }

  /// Netlist
  auto idb_net_list = _design->get_net_list();
  for (auto net : idb_net_list->get_net_list()) {
    alignRect(net->get_bounding_box());
    alignCoord(net->get_average_coordinate());

    auto wire_list = net->get_wire_list();
    for (auto wire : wire_list->get_wire_list()) {
      for (auto segment : wire->get_segment_list()) {
        alignSignalSegment(segment);
      }
    }
  }

  /// special net list
  auto idb_pdn_list = _design->get_special_net_list();
  for (auto special_net : idb_pdn_list->get_net_list()) {
    auto wire_list = special_net->get_wire_list();
    for (auto wire : wire_list->get_wire_list()) {
      for (auto segment : wire->get_segment_list()) {
        alignSpecialSegment(segment);
      }
    }
  }

  /// via list
  auto idb_via_list = _design->get_via_list();
  for (auto idb_via : idb_via_list->get_via_list()) {
    alignVia(idb_via);
  }

  /// Blockage list
  auto idb_blk_list = _design->get_blockage_list();
  for (auto idb_blockage : idb_blk_list->get_blockage_list()) {
    for (auto rect : idb_blockage->get_rect_list()) {
      alignRect(rect);
    }
  }

  /// Region list
  auto idb_region_list = _design->get_region_list();
  for (auto idb_region : idb_region_list->get_region_list()) {
    for (auto rect : idb_region->get_boundary()) {
      alignRect(rect);
    }
  }

  /// slot list
  auto idb_slot_list = _design->get_slot_list();
  for (auto idb_slot : idb_slot_list->get_slot_list()) {
    for (auto rect : idb_slot->get_rect_list()) {
      alignRect(rect);
    }
  }

  /// fill list
  auto idb_fill_list = _design->get_fill_list();
  for (auto idb_fill : idb_fill_list->get_fill_list()) {
    /// fill layer
    auto fill_layer = idb_fill->get_layer();
    for (auto rect : fill_layer->get_rect_list()) {
      alignRect(rect);
    }

    /// fill via
    auto fill_via = idb_fill->get_via();
    alignVia(fill_via->get_via());
    for (auto fill_coord : fill_via->get_coordinate_list()) {
      alignCoord(fill_coord);
    }
  }

  /// die
  alignRect(idb_die->get_bounding_box());
  for (auto pt : idb_die->get_points()) {
    alignCoord(pt);
  }

  return true;
}

}  // namespace idm
