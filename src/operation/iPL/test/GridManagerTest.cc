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
/*
 * @Author: S.J Chen
 * @Date: 2022-03-01 16:19:46
 * @LastEditTime: 2022-04-06 14:37:35
 * @LastEditors: S.J Chen
 * @Description:
 * @FilePath: /iEDA/src/iPL/test/GridManagerTest.cc
 * Contact : https://github.com/sjchanson
 */

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "module/grid_manager/GridManager.hh"
#include "module/logger/Log.hh"

namespace ipl {

class GridManagerTest : public testing::Test
{
  void SetUp()
  {
    char config[] = "gridmanager test";
    char* argv[] = {config};
    Log::init(argv);
  }
  void TearDown() { Log::end(); }
};

TEST_F(GridManagerTest, sample)
{
  GridManager* grid_manager = new GridManager(Rectangle<int32_t>(0, 0, 12, 12), 3, 3, 1.0);

  LOG_INFO << "Grid X Cnt : " << grid_manager->get_grid_cnt_x();
  LOG_INFO << "Row Y Cnt : " << grid_manager->get_grid_cnt_y();

  LOG_INFO << "Add Row/Site Rectangle";
  Rectangle<int32_t> rect1(4, 0, 8, 4);
  Rectangle<int32_t> rect2(4, 4, 12, 8);
  std::vector<Grid*> overlap_grid_list1;
  grid_manager->obtainOverlapGridList(overlap_grid_list1, rect1);
  for (auto* grid : overlap_grid_list1) {
    int32_t overlap_area = obtainOverlapArea(grid, rect1);
    grid->add_area(overlap_area);
  }
  std::vector<Grid*> overlap_grid_list2;
  grid_manager->obtainOverlapGridList(overlap_grid_list2, rect2);
  for (auto* grid : overlap_grid_list2) {
    int32_t overlap_area = obtainOverlapArea(grid, rect2);
    grid->add_area(overlap_area);
  }

  LOG_INFO << "Print the Available Grids in each Row";
  for (auto* row : grid_manager->get_row_list()) {
    LOG_INFO << "ROW : " << row->get_row_idx();
    LOG_INFO << "AVAILABLE SITE : ";
    for (auto* grid : row->obtainAvailableGridList(1.0)) {
      LOG_INFO << "INDEX : " << grid->get_grid_idx();
    }
    LOG_INFO << "AVAILABLE RANGE : ";
    for (auto rect : row->obtainAvailableGridShapeList(1.0)) {
      LOG_INFO << rect.get_ll_x() << "," << rect.get_ll_y() << " " << rect.get_ur_x() << "," << rect.get_ur_y();
    }
  }

  LOG_INFO << "Identify if there is overlap";
  Rectangle<int32_t> rect3(2, 4, 6, 8);
  std::vector<Grid*> overlap_grid_list3;
  grid_manager->obtainOverlapGridList(overlap_grid_list3, rect3);
  for (auto* grid : overlap_grid_list3) {
    int32_t overlap_area = obtainOverlapArea(grid, rect3);
    grid->add_area(overlap_area);
  }
  std::vector<Grid*> overflow_grid_list = grid_manager->obtainOverflowIllegalGridList();
  for (auto* grid : overflow_grid_list) {
    LOG_INFO << "OVERLAP GRID : " << grid->get_row_idx() << "," << grid->get_grid_idx();
  }

  LOG_INFO << "Clear All Grid Occupied Area";
  grid_manager->clearAllOccupiedArea();

  delete grid_manager;
}

}  // namespace ipl
