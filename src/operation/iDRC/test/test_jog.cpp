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
#include "JogSpacingCheck.hpp"
#include "idm.h"
#include "ids.hpp"

using namespace idrc;

void setRectValue(ids::DRCRect& rect, int lb_x, int lb_y, int rt_x, int rt_y, std::string name, int so_id)
{
  rect.lb_x = lb_x;
  rect.lb_y = lb_y;
  rect.rt_x = rt_x;
  rect.rt_y = rt_y;
  rect.layer_name = name;
  rect.so_id = so_id;
  rect.type = ids::RectType::kRouting;
}

int main(int argc, char* argv[])
{
  std::string path = "";
  dmInst->get_config().set_tech_lef_path(path);
  vector<string> path_list;
  path_list.push_back(path);
  dmInst->readLef(path_list);
  std::cout << "read finish" << std::endl;

  DrcAPIInst.initDRC();

  ids::DRCRect rect1;
  setRectValue(rect1, 1000, 1000, 2000, 1600, "M1", 0);
  ids::DRCRect rect2;
  setRectValue(rect2, 1000, 2100, 1701, 2200, "M1", 0);
  ids::DRCRect rect3;
  setRectValue(rect3, 1000, , 1320, 1280, "M1", 0);
  ids::DRCEnv env;
  // env.push_back(rect1);
  // env.push_back(rect2);
  // env.push_back(rect3);
  ids::DRCTask task;
  task.push_back(rect1);
  task.push_back(rect2);
  task.push_back(rect3);
  if (DrcAPIInst.checkDRC(env, task)) {
    std::cout << "true" << std::endl;
  } else {
    std::cout << "false" << std::endl;
  }

  return 0;
}