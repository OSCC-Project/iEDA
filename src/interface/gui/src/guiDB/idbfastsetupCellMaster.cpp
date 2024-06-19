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
#include "IdbCellMaster.h"
#include "IdbInstance.h"
#include "feature_manager.h"
#include "guiConfig.h"
#include "idbfastsetup.h"

void IdbSpeedUpSetup::showCellMasters() {
  auto route_data = featureInst->get_route_data();

  /// to reuse api of gui instance, build instance list for cell master first
  IdbInstanceList idb_inst_list;

  int origin_x                   = _layout->get_core()->get_bounding_box()->get_low_x();
  int origin_y                   = _layout->get_core()->get_bounding_box()->get_low_y();
  int max_y                      = 0;
  IdbCellMasterList* master_list = _layout->get_cell_master_list();

  for (auto [cell_master_name, cell_master] : route_data.cell_master_list) {
    /// new a instance
    IdbInstance* idb_inst = idb_inst_list.add_instance(cell_master_name);
    auto idb_cell_master  = master_list->find_cell_master(cell_master_name);
    if (idb_cell_master == nullptr) {
      std::cout << "Error can not find Cell Master : " << cell_master_name << std::endl;
      continue;
    }
    idb_inst->set_cell_master(idb_cell_master);
    idb_inst->set_status_placed();
    idb_inst->set_orient(IdbOrient::kN_R0, false);
    idb_inst->set_coodinate(origin_x, origin_y);

    /// build pa data to term
    for (auto [term_name, term_pa] : cell_master.term_list) {
      auto idb_term = idb_cell_master->findTerm(term_name);
      if (idb_term == nullptr) {
        std::cout << "Error can not find term : " << term_name << std::endl;
        continue;
      }

      auto& pa_list = idb_term->get_pa_list();
      for (auto db_pa : term_pa.pa_list) {
        IdbCoordinate<int32_t>* coord = new IdbCoordinate<int32_t>(db_pa.x + origin_x, db_pa.y + origin_y);
        pa_list.push_back(coord);
      }
    }

    // update origin if reach right end of core
    origin_x = origin_x + idb_inst->get_bounding_box()->get_width();

    if (max_y < origin_y + idb_inst->get_bounding_box()->get_height()) {
      max_y = origin_y + idb_inst->get_bounding_box()->get_height();
    }

    if (origin_x > _layout->get_core()->get_bounding_box()->get_width() / 2) {
      origin_x = _layout->get_core()->get_bounding_box()->get_low_x();
      origin_y = max_y;
    }
  }

  /// build instance list
  createInstance(&idb_inst_list);
}
