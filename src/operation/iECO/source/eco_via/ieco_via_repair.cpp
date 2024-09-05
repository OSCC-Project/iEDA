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
#include "ieco_via_repair.h"

#include "IdbLayerShape.h"
#include "IdbViaMaster.h"
#include "IdbVias.h"
#include "ieco_dm.h"
#include "ieco_via_init.h"
#include "omp.h"

namespace ieco {

ECOViaRepair::ECOViaRepair(EcoDataManager* data_manager)
{
  _data_manager = data_manager;
}

ECOViaRepair::~ECOViaRepair()
{
}

int ECOViaRepair::repairByShape()
{
  int repair_num = 0;
  int total = 0;

  for (auto& [layer, via_layer] : _data_manager->get_eco_data().get_via_layers()) {
    auto via_masters = via_layer.get_default();
    auto& via_instances = via_layer.get_via_instances();

    std::vector<bool> states(via_instances.size(), false);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int) via_instances.size(); ++i) {
      states[i] = changeViaMaster(via_instances[i], via_masters);
    }

    int number = 0;
    for (auto state : states) {
      if (state) {
        repair_num++;
        number++;
      }
    }

    total += via_instances.size();

    std::cout << "iECO : change via num = " << number << " / " << via_instances.size() << " in layer " << layer << std::endl;
  }

  std::cout << "iECO : change via num = " << repair_num << " / " << total << std::endl;

  return repair_num;
}

int ECOViaRepair::repairByPattern()
{
  int repair_num = 0;

  return repair_num;
}

bool ECOViaRepair::changeViaMaster(EcoDataVia& via, std::vector<EcoDataViaMaster>& via_masters)
{
  bool b_changed = false;

  if (via.is_connect_bottom_empty() && via.is_connect_top_empty()) {
    return b_changed;
  }

  auto move_rect = [](idb::IdbRect* master_rect, int bias_x, int bias_y) -> idb::IdbRect {
    idb::IdbRect rect(master_rect);
    rect.moveByStep(bias_x, bias_y);

    return rect;
  };

  auto is_same_direction = [](Direction direction, idb::IdbLayer* layer) -> bool {
    auto routing_layer = dynamic_cast<idb::IdbLayerRouting*>(layer);
    if (direction >= Direction::kNorth && direction <= Direction::kVertical
        && routing_layer->get_direction() == idb::IdbLayerDirection::kVertical) {
      return true;
    }

    if (direction >= Direction::kWest && direction <= Direction::kHorizontal
        && routing_layer->get_direction() == idb::IdbLayerDirection::kHorizontal) {
      return true;
    }

    return false;
  };

  auto idb_via = via.get_idb_via();
  int via_x = idb_via->get_coordinate()->get_x();
  int via_y = idb_via->get_coordinate()->get_y();

  int64_t min_area_top = via.get_connect_top().getMergeRectArea(
      idb_via->get_top_layer_shape().get_rect(0)->get_low_x(), idb_via->get_top_layer_shape().get_rect(0)->get_low_y(),
      idb_via->get_top_layer_shape().get_rect(0)->get_high_x(), idb_via->get_top_layer_shape().get_rect(0)->get_high_y());
  int64_t min_area_bottom = via.get_connect_bottom().getMergeRectArea(
      idb_via->get_bottom_layer_shape().get_rect(0)->get_low_x(), idb_via->get_bottom_layer_shape().get_rect(0)->get_low_y(),
      idb_via->get_bottom_layer_shape().get_rect(0)->get_high_x(), idb_via->get_bottom_layer_shape().get_rect(0)->get_high_y());
  int64_t min_area = min_area_top + min_area_bottom;
  //   int64_t min_area = INT64_MAX;
  idb::IdbViaMaster* master_select = nullptr;

  for (auto& master : via_masters) {
    if (master.get_idb_via_master() == nullptr || master.get_idb_via_master() == idb_via->get_instance()) {
      continue;
    }

    bool top_matched = false;
    bool bottom_matched = false;
    int64_t top_area = 0;
    int64_t bottom_area = 0;

    if (false == via.is_connect_top_empty()) {
      auto* top_rect = master.get_idb_via_master()->get_top_layer_shape()->get_rect(0);
      auto top_try = move_rect(top_rect, via_x, via_y);

      top_area
          = via.get_connect_top().getMergeRectArea(top_try.get_low_x(), top_try.get_low_y(), top_try.get_high_x(), top_try.get_high_y());
      if (top_area <= min_area_top) {
        top_matched = true;
      }
    }

    if (false == via.is_connect_bottom_empty()) {
      auto* bottom_rect = master.get_idb_via_master()->get_bottom_layer_shape()->get_rect(0);
      auto bottom_try = move_rect(bottom_rect, via_x, via_y);
      bottom_area = via.get_connect_bottom().getMergeRectArea(bottom_try.get_low_x(), bottom_try.get_low_y(), bottom_try.get_high_x(),
                                                              bottom_try.get_high_y());
      if (bottom_area <= min_area_bottom) {
        bottom_matched = true;
      }
    }

    if (top_matched && bottom_matched) {
      // if ((top_matched && bottom_matched) || (top_matched && via.is_connect_bottom_empty())
      //     || (bottom_matched && via.is_connect_top_empty())) {
      // if ((top_matched && bottom_matched) || (bottom_matched && (top_area + bottom_area) < min_area)) {
      // || (bottom_matched && via.is_bottom_connected_pin()
      //     && is_same_direction(master.get_info().top_direction, via.get_idb_via()->get_top_layer_shape().get_layer()))) {
      min_area_top = top_area;
      min_area_bottom = bottom_area;
      min_area = top_area + bottom_area;

      master_select = master.get_idb_via_master();
    }
  }

  b_changed = master_select == nullptr || master_select->get_name() == idb_via->get_instance()->get_name() ? false : true;

  /// reset via master
  if (b_changed) {
    idb_via->reset_instance(master_select);
  }

  return b_changed;
}

}  // namespace ieco