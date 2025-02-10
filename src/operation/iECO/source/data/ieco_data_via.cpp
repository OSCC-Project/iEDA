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
#include "ieco_data_via.h"

#include "IdbViaMaster.h"
#include "IdbVias.h"

namespace ieco {

EcoDataViaMaster::EcoDataViaMaster(idb::IdbViaMaster* idb_via_master)
{
  _idb_via_master = idb_via_master;

  initViaInfo();
}

EcoDataViaMaster::~EcoDataViaMaster()
{
}
/// @brief  only support via with 1 row * 1 col cuts
void EcoDataViaMaster::initViaInfo()
{
  auto getEnclosureDirection = [](idb::IdbRect* cut_rect, idb::IdbRect* enclosure_rect) -> Direction {
    Direction direction;

    auto center_x = cut_rect->get_middle_point_x();
    auto center_y = cut_rect->get_middle_point_y();
    auto enclosure_witdh = enclosure_rect->get_width();
    auto enclosre_height = enclosure_rect->get_height();

    if (enclosure_witdh > enclosre_height) {
      /// horizontal
      auto delta_x_west = center_x - enclosure_rect->get_low_x();
      auto delta_x_east = enclosure_rect->get_high_x() - center_x;
      direction = delta_x_west == delta_x_east ? Direction::kMiddleHorizontal
                                               : (delta_x_west > delta_x_east ? Direction::kWest : Direction::kEast);

    } else if (enclosure_witdh < enclosre_height) {
      /// vertical
      auto delta_y_south = center_y - enclosure_rect->get_low_y();
      auto delta_y_north = enclosure_rect->get_high_y() - center_y;
      direction = delta_y_south == delta_y_north ? Direction::kMiddleVertical
                                                 : (delta_y_south > delta_y_north ? Direction::kSouth : Direction::kNorth);
    } else {
      /// middle
      direction = Direction::kMiddle;  // cut in via center
    }

    return direction;
  };

  /// only support via with 1 row * 1 col cuts
  if (_idb_via_master != nullptr && 1 != _idb_via_master->get_cut_layer_shape()->get_rect_list_num()) {
    return;
  }
  //   _info.cols = _idb_via_master->get_cut_cols();
  //   _info.rows = _idb_via_master->get_cut_rows();
  _info.cols = 1;
  _info.rows = 1;

  auto* bottom_rect = _idb_via_master->get_bottom_layer_shape()->get_rect(0);
  auto* cut_rect = _idb_via_master->get_cut_layer_shape()->get_rect(0);
  auto* top_rect = _idb_via_master->get_top_layer_shape()->get_rect(0);
  _info.bottom_shape.addRect(bottom_rect->get_low_x(), bottom_rect->get_low_y(), bottom_rect->get_high_x(), bottom_rect->get_high_y());
  _info.top_shape.addRect(top_rect->get_low_x(), top_rect->get_low_y(), top_rect->get_high_x(), top_rect->get_high_y());
  _info.bottom_direction = getEnclosureDirection(cut_rect, bottom_rect);
  _info.top_direction = getEnclosureDirection(cut_rect, top_rect);
}

bool EcoDataViaMaster::isDefault()
{
  auto* bottom_shape = _idb_via_master->get_bottom_layer_shape();
  auto bottom_layer_width = dynamic_cast<idb::IdbLayerRouting*>(bottom_shape->get_layer())->get_width();
  auto* bottom_rect = bottom_shape->get_rect(0);
  if (bottom_rect->get_width() > bottom_layer_width && bottom_rect->get_height() > bottom_layer_width) {
    return false;
  }

  auto top_shape = _idb_via_master->get_top_layer_shape();
  auto top_layer_width = dynamic_cast<idb::IdbLayerRouting*>(top_shape->get_layer())->get_width();
  auto* top_rect = top_shape->get_rect(0);
  if (top_rect->get_width() > top_layer_width && top_rect->get_height() > top_layer_width) {
    return false;
  }

  //   return _idb_via_master->is_default() && _idb_via_master->isOneCut();
  return _idb_via_master->isOneCut();
}

bool EcoDataViaMaster::isMatchRowsCols(int row_num, int col_num)
{
  return _info.rows == row_num && _info.cols == col_num;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

EcoDataVia::EcoDataVia(idb::IdbVia* idb_via)
{
  _idb_via = idb_via;
}

EcoDataVia::~EcoDataVia()
{
}

bool EcoDataVia::is_connect_top_empty()
{
  return _connect_top.get_polyset().empty();
}

bool EcoDataVia::is_connect_bottom_empty()
{
  return _connect_bottom.get_polyset().empty();
}

/// check if intersected with top enclosure
bool EcoDataVia::intersectedTop(int llx, int lly, int urx, int ury)
{
  auto top_shape = _idb_via->get_top_layer_shape();
  return top_shape.isIntersected(llx, lly, urx, ury);
}
/// check if intersected with bottom enclosure
bool EcoDataVia::intersectedBottom(int llx, int lly, int urx, int ury)
{
  auto bottom_shape = _idb_via->get_bottom_layer_shape();
  return bottom_shape.isIntersected(llx, lly, urx, ury);
}
/// add rect which connected to top enclosure
void EcoDataVia::addConnectedTop(int llx, int lly, int urx, int ury)
{
  _connect_top.addRect(llx, lly, urx, ury);
}
/// add rect which connected to bottom enclosure
void EcoDataVia::addConnectedBottom(int llx, int lly, int urx, int ury)
{
  _connect_bottom.addRect(llx, lly, urx, ury);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
EcoDataViaLayer::EcoDataViaLayer()
{
}

EcoDataViaLayer::~EcoDataViaLayer()
{
  _via_masters.clear();
  _via_instances.clear();
  std::vector<EcoDataVia>().swap(_via_instances);
}

std::vector<EcoDataViaMaster> EcoDataViaLayer::get_default()
{
  std::vector<EcoDataViaMaster> defautl_masters;
  for (auto& [name, via_master] : _via_masters) {
    if (via_master.isDefault()) {
      defautl_masters.emplace_back(via_master);
    }
  }

  return defautl_masters;
}

void EcoDataViaLayer::addViaMaster(std::string master_name, idb::IdbViaMaster* idb_via_master)
{
  _via_masters.insert(std::make_pair(master_name, EcoDataViaMaster(idb_via_master)));
}

void EcoDataViaLayer::addVia(idb::IdbVia* idb_via)
{
  EcoDataVia eco_via(idb_via);
  _via_instances.emplace_back(EcoDataVia(idb_via));
}

void EcoDataViaLayer::addVia(EcoDataVia eco_via)
{
  _via_instances.emplace_back(eco_via);
}

}  // namespace ieco