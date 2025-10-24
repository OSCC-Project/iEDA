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
#include "guiConfig.h"
#include "idbfastsetup.h"
#include "omp.h"

void IdbSpeedUpSetup::showDrc(std::map<std::string, std::map<std::string, std::vector<ids::Violation>>>& drc_db,
                              int max_num) {
  for (auto& [rule_type, drc_layers] : drc_db) {
    // std::string rule = rule_type;
    // if (rule == "Default Spacing") {
    //   rule = "ParallelRunLength Spacing";
    // }

    auto container = _gui_design->get_drc_container(rule_type);
    if (container != nullptr) {
      int DRC_MAX = max_num > 0 ? max_num : INT32_MAX;
      for (auto& [layer, violations] : drc_layers) {
        auto drc_list = container->findDrcList(layer);

        for (auto violation : violations) {
          if (DRC_MAX == 0) {
            return;
          }

          /// process violation
          createDrc(drc_list, violation);

          DRC_MAX--;
        }
      }
    } else {
      std::cout << "find no drc container view : " << rule_type << std::endl;
    }
  }
}

void IdbSpeedUpSetup::createDrc(GuiSpeedupDrcList* drc_list, ids::Violation& drc_db) {
  if (drc_list == nullptr) {
    return;
  }

  int min_x = drc_db.ll_x;
  int min_y = drc_db.ll_y;
  int max_x = drc_db.ur_x;
  int max_y = drc_db.ur_y;

  if (min_x == max_x) {
    max_x += 2;
  }

  if (min_y == max_y) {
    max_y += 2;
  }

  /// if line
  if (min_x == max_x || min_y == max_y) {
    qreal q_min_x       = _transform.db_to_guidb(min_x);
    qreal q_min_y       = _transform.db_to_guidb_rotate(min_y);
    qreal q_max_x       = _transform.db_to_guidb(max_x);
    qreal q_max_y       = _transform.db_to_guidb_rotate(max_y);
    GuiSpeedupDrc* item = drc_list->findItem(QPointF((q_min_x + q_max_x) / 2, (q_min_y + q_max_y) / 2));
    if (item == nullptr) {
      std::cout << "Error : can not find Drc item in die" << std::endl;
      return;
    }

    item->add_point(QPointF(q_min_x, q_min_y), QPointF(q_max_x, q_max_y));
  } else {
    /// rect
    QRectF rect         = _transform.db_to_guidb_rect(min_x, min_y, max_x, max_y);
    GuiSpeedupDrc* item = drc_list->findItem(rect.center());
    if (item == nullptr) {
      std::cout << "Error : can not find Drc item in die" << std::endl;
      return;
    }
    item->add_rect(rect);
  }
}
