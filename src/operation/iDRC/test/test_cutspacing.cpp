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
#include <string>

#include "DRC.h"
#include "DrcAPI.hpp"
#include "DrcRect.h"
#include "idm.h"
#include "ids.hpp"

using namespace idrc;

// void setRectValue(ids::DRCRect& rect, int lb_x, int lb_y, int rt_x, int rt_y, std::string name, int so_id)
// {
//   rect.lb_x = lb_x;
//   rect.lb_y = lb_y;
//   rect.rt_x = rt_x;
//   rect.rt_y = rt_y;
//   rect.layer_name = name;
//   rect.so_id = so_id;
//   rect.type = ids::RectType::kRouting;
// }

int main(int argc, char* argv[])
{
  std::string path = "";
  dmInst->get_config().set_tech_lef_path(path);
  vector<string> path_list;
  path_list.push_back(path);
  dmInst->readLef(path_list);
  std::cout << "read finish" << std::endl;

  DrcAPIInst.initDRC();

  DrcRect* rect1 = DrcAPIInst.getDrcRect(1, RectOwnerType::kViaCut, 100, 100, 200, 200, "VIA7");
  DrcRect* rect2 = DrcAPIInst.getDrcRect(7, RectOwnerType::kViaCut, 200, 200, 300, 300, "VIA7");
  std::vector<DrcRect*> origin_rect_list{rect1, rect2};

  if (DrcAPIInst.checkDRC(origin_rect_list)) {
    std::cout << "true" << std::endl;
  } else {
    std::cout << "false" << std::endl;
  }
  // for (auto& rect : scope_list) {
  //   std::cout << "min_corner:" << rect->get_left() << "," << rect->get_bottom() << " max_corner:" << rect->get_right() << ","
  //             << rect->get_top() << std::endl;
  // }
  //   if (DrcAPIInst.getMaxScope(origin_rect_list)) {
  //     std::cout << "true" << std::endl;
  //   } else {
  //     std::cout << "false" << std::endl;
  //   }

  return 0;
}