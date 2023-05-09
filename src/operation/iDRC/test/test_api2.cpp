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

#include "DrcAPI.hpp"
#include "DrcRect.h"
#include "RegionQuery.h"
#include "idm.h"

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
  ids::DRCTask task;
  std::vector<ids::DRCTask> task_list;
  task.region_query = DrcAPIInst.init();
  DrcRect* rect1 = new DrcRect(1, 1000, 1000, 1100, 2000, 1);
  DrcRect* rect2 = new DrcRect(1, 1000, 1900, 2000, 2000, 1);
  DrcRect* rect3 = new DrcRect(1, 1900, 1000, 2000, 2000, 1);

  DrcRect* rect4 = new DrcRect(1, 3000, 1000, 3100, 2000, 1);
  DrcRect* rect5 = new DrcRect(1, 3000, 1900, 4000, 2000, 1);

  DrcRect* rect6 = new DrcRect(1, 5000, 1000, 5100, 2000, 1);

  DrcRect* rect7 = new DrcRect(1, 1950, 1000, 2050, 2000, 1);
  DrcRect* rect8 = new DrcRect(1, 2250, 1000, 3150, 1100, 1);

  task.drc_rect_list.push_back(rect1);
  task.drc_rect_list.push_back(rect2);
  task.drc_rect_list.push_back(rect3);
  // task.drc_rect_list.push_back(rect4);
  // task.drc_rect_list.push_back(rect5);
  // task.drc_rect_list.push_back(rect6);
  task_list.push_back(task);
  ids::DRCTask task1;
  task1.region_query = task.region_query;
  task1.drc_rect_list.push_back(rect4);
  task1.drc_rect_list.push_back(rect5);
  task_list.push_back(task1);
  ids::DRCTask task2;
  task2.region_query = task.region_query;
  task2.drc_rect_list.push_back(rect6);
  task_list.push_back(task2);
  DrcAPIInst.add(task_list);
  auto polys = task.region_query->getPolys(1, 1);
  // for (auto poly : task.region_query->getPolys(1, 1)) {
  //   for (auto scope : poly->getScopes()) {
  //     std::cout << scope->get_left() << std::endl;
  //     std::cout << scope->get_bottom() << std::endl;
  //     std::cout << scope->get_right() << std::endl;
  //     std::cout << scope->get_top() << std::endl;
  //     std::cout << "################################" << std::endl;
  //   }
  //   std::cout << "*********************************************" << std::endl;
  // }
  // std::cout << "del" << std::endl;

  std::vector<ids::DRCTask> task_list1;
  ids::DRCTask task4;
  task4.region_query = task.region_query;
  task4.drc_rect_list.push_back(rect2);
  task4.drc_rect_list.push_back(rect5);
  task4.drc_rect_list.push_back(rect8);
  task_list1.push_back(task4);

  DrcAPIInst.del(task_list1);
  std::cout << "del finish" << std::endl;
  std::cout << task.region_query->getPolys(1, 1).size() << std::endl;
  if (DrcAPIInst.check(task_list1)) {
    std::cout << "true" << std::endl;
  } else {
    std::cout << "false" << std::endl;
  }

  // for (auto poly : task.region_query->getPolys(1, 1)) {
  //   for (auto scope : poly->getScopes()) {
  //     std::cout << scope->get_left() << std::endl;
  //     std::cout << scope->get_bottom() << std::endl;
  //     std::cout << scope->get_right() << std::endl;
  //     std::cout << scope->get_top() << std::endl;
  //     std::cout << "################################" << std::endl;
  //   }
  //   std::cout << "*********************************************" << std::endl;
  // }
  return 0;
}