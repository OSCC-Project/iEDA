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
 * @file		feature_builder.h
 * @date		13/05/2024
 * @version		0.1
 * @description


        build feature data
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "route_builder.h"

#include "IdbPins.h"
#include "IdbVias.h"
#include "RTInterface.hpp"
#include "idm.h"

namespace ieda_feature {

bool RouteDataBuilder::buildRouteData()
{
  auto idb_design = dmInst->get_idb_design();
  auto net_list = idb_design->get_net_list();

  for (IdbNet* net : net_list->get_net_list()) {
    auto pin_list = net->get_instance_pin_list();

    for (IdbRegularWire* wire : net->get_wire_list()->get_wire_list()) {
      for (IdbRegularWireSegment* segment : wire->get_segment_list()) {
        if (segment->is_via()) {
          for (auto via : segment->get_via_list()) {
            for (auto pin : pin_list->get_pin_list()) {
              if (is_pa(pin, via)) {
                auto instance = pin->get_instance();
                auto cell_master_name = instance->get_cell_master()->get_name();
                auto term_name = pin->get_term_name();
                CellMasterPA pa;
                pa.name = cell_master_name;

                TermPA& term_pa = find_cell_term_pa(cell_master_name, term_name);
                DbPinAccess pa_db;

                IdbOrientTransform db_transform(instance->get_orient(), instance->get_coordinate(),
                                                instance->get_cell_master()->get_width(), instance->get_cell_master()->get_height());

                auto pa_coord = new IdbCoordinate<int32_t>(via->get_coordinate()->get_x(), via->get_coordinate()->get_y());
                db_transform.cellMasterCoordinate(pa_coord);

                pa_db.layer = via->get_bottom_layer_shape().get_layer()->get_name();
                pa_db.x = pa_coord->get_x();
                pa_db.y = pa_coord->get_y();

                delete pa_coord;

                add_term_pa(term_pa, pa_db);
              }
            }
          }
        }
      }
    }
  }

  return true;
}

bool RouteDataBuilder::is_pa(IdbPin* pin, IdbVia* via)
{
  if (pin == nullptr || via == nullptr) {
    return false;
  }

  if (pin->isIntersected(via->get_coordinate()->get_x(), via->get_coordinate()->get_y(), via->get_bottom_layer_shape().get_layer())) {
    return true;
  }

  return false;
}

void RouteDataBuilder::add_term_pa(TermPA& term_pa, DbPinAccess pin_access)
{
  for (auto& pa : term_pa.pa_list) {
    if (pa.layer == pin_access.layer && pa.x == pin_access.x && pa.y == pin_access.y) {
      pa.number += 1;
      return;  /// exist
    }
  }

  pin_access.number += 1;
  term_pa.pa_list.push_back(pin_access);
}

TermPA& RouteDataBuilder::find_cell_term_pa(std::string cell_master_name, std::string term_name)
{
  /// check exist
  auto cell_master = _data->cell_master_list.find(cell_master_name);
  if (cell_master != _data->cell_master_list.end()) {
    auto term_pa = cell_master->second.term_list.find(term_name);
    if (term_pa != cell_master->second.term_list.end()) {
      return term_pa->second;
    } else {
      TermPA term_pa;
      cell_master->second.term_list.insert(std::make_pair(term_name, term_pa));

      return _data->cell_master_list[cell_master_name].term_list[term_name];
    }
  } else {
    CellMasterPA master_pa;
    master_pa.name = cell_master_name;

    TermPA term_pa;
    master_pa.term_list.insert(std::make_pair(term_name, term_pa));

    _data->cell_master_list.insert(std::make_pair(cell_master_name, master_pa));

    return _data->cell_master_list[cell_master_name].term_list[term_name];
  }

  /// add pa
}

}  // namespace ieda_feature