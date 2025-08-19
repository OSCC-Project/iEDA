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

void IdbSpeedUpSetup::showDrc(std::map<std::string, std::vector<idrc::DrcViolation*>>& drc_db, int max_num) {
  for (auto [rule_type, drc_spot_list] : drc_db) {
    std::cout << "drc rule=" << rule_type << " number=" << drc_spot_list.size() << std::endl;

    for (auto [rule_type, drc_spot_list] : drc_db) {
      std::string rule = rule_type;
      if (rule == "Default Spacing") {
        rule = "ParallelRunLength Spacing";
      }

      auto container = _gui_design->get_drc_container(rule);
      if (container != nullptr) {
        if (max_num > 0) {
          int size = max_num > drc_spot_list.size() ? drc_spot_list.size() : max_num;
          // #pragma omp parallel for
          for (int i = 0; i < size; i++) {
            /// create drc view
            std::string layer = drc_spot_list[i]->get_layer()->get_name();
            auto drc_list     = container->findDrcList(layer);
            createDrc(drc_list, drc_spot_list[i]);
          }
        } else {
          // #pragma omp parallel for
          for (auto drc_spot : drc_spot_list) {
            /// create drc view
            std::string layer = drc_spot->get_layer()->get_name();
            auto drc_list     = container->findDrcList(layer);
            createDrc(drc_list, drc_spot);
          }
        }
      } else {
        std::cout << "find no drc container view : " << rule << std::endl;
      }
    }
  }
}

void IdbSpeedUpSetup::createDrc(GuiSpeedupDrcList* drc_list, idrc::DrcViolation* drc_db) {
  if (drc_list == nullptr || drc_db == nullptr) {
    return;
  }

  auto* spot_rect = static_cast<idrc::DrcViolationRect*>(drc_db);
  int min_x       = spot_rect->get_llx();
  int min_y       = spot_rect->get_lly();
  int max_x       = spot_rect->get_urx();
  int max_y       = spot_rect->get_ury();

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
