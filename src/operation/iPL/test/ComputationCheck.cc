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
 * @Date: 2022-02-25 17:44:51
 * @LastEditTime: 2022-10-31 20:22:38
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/test/ComputationCheck.cc
 * Contact : https://github.com/sjchanson
 */

#include "data/Instance.hh"
#include "gtest/gtest.h"
#include "module/logger/Log.hh"

namespace ipl {

class ComputationCheck : public testing::Test
{
};

TEST_F(ComputationCheck, instance_orient)
{
  Instance* inst = new Instance("test_inst");

  // must set the cell master.
  Cell* cell_master = new Cell("test_master");
  cell_master->set_width(4);
  cell_master->set_height(8);
  inst->set_cell_master(cell_master);

  inst->set_orient(Orient::kN_R0);
  inst->set_shape(0, 0, 4, 8);
  Pin* pin0 = new Pin("pin0");
  pin0->set_offset_coordi(1, 1);
  inst->add_pin(pin0);
  Pin* pin1 = new Pin("pin1");
  pin1->set_offset_coordi(-1, 2);
  inst->add_pin(pin1);
  inst->update_coordi(0, 0);

  int32_t x, y;
  x = pin0->get_center_coordi().get_x();
  y = pin0->get_center_coordi().get_y();
  EXPECT_EQ(3, x);
  EXPECT_EQ(5, y);
  x = pin1->get_center_coordi().get_x();
  y = pin1->get_center_coordi().get_y();
  EXPECT_EQ(1, x);
  EXPECT_EQ(6, y);

  // orient W
  inst->set_orient(Orient::kW_R90);
  x = pin0->get_center_coordi().get_x();
  y = pin0->get_center_coordi().get_y();
  EXPECT_EQ(3, x);
  EXPECT_EQ(3, y);
  x = pin1->get_center_coordi().get_x();
  y = pin1->get_center_coordi().get_y();
  EXPECT_EQ(2, x);
  EXPECT_EQ(1, y);

  // orient S
  inst->set_orient(Orient::kS_R180);
  x = pin0->get_center_coordi().get_x();
  y = pin0->get_center_coordi().get_y();
  EXPECT_EQ(1, x);
  EXPECT_EQ(3, y);
  x = pin1->get_center_coordi().get_x();
  y = pin1->get_center_coordi().get_y();
  EXPECT_EQ(3, x);
  EXPECT_EQ(2, y);

  // orient E
  inst->set_orient(Orient::kE_R270);
  x = pin0->get_center_coordi().get_x();
  y = pin0->get_center_coordi().get_y();
  EXPECT_EQ(5, x);
  EXPECT_EQ(1, y);
  x = pin1->get_center_coordi().get_x();
  y = pin1->get_center_coordi().get_y();
  EXPECT_EQ(6, x);
  EXPECT_EQ(3, y);

  // orient FN
  inst->set_orient(Orient::kFN_MY);
  x = pin0->get_center_coordi().get_x();
  y = pin0->get_center_coordi().get_y();
  EXPECT_EQ(1, x);
  EXPECT_EQ(5, y);
  x = pin1->get_center_coordi().get_x();
  y = pin1->get_center_coordi().get_y();
  EXPECT_EQ(3, x);
  EXPECT_EQ(6, y);

  // orient FS
  inst->set_orient(Orient::kFS_MX);
  x = pin0->get_center_coordi().get_x();
  y = pin0->get_center_coordi().get_y();
  EXPECT_EQ(3, x);
  EXPECT_EQ(3, y);
  x = pin1->get_center_coordi().get_x();
  y = pin1->get_center_coordi().get_y();
  EXPECT_EQ(1, x);
  EXPECT_EQ(2, y);

  // orient FE
  inst->set_orient(Orient::kFE_MY90);
  x = pin0->get_center_coordi().get_x();
  y = pin0->get_center_coordi().get_y();
  EXPECT_EQ(3, x);
  EXPECT_EQ(1, y);
  x = pin1->get_center_coordi().get_x();
  y = pin1->get_center_coordi().get_y();
  EXPECT_EQ(2, x);
  EXPECT_EQ(3, y);

  // orient FW
  inst->set_orient(Orient::kFW_MX90);
  x = pin0->get_center_coordi().get_x();
  y = pin0->get_center_coordi().get_y();
  EXPECT_EQ(5, x);
  EXPECT_EQ(3, y);
  x = pin1->get_center_coordi().get_x();
  y = pin1->get_center_coordi().get_y();
  EXPECT_EQ(6, x);
  EXPECT_EQ(1, y);

  // change coordinate of inst
  inst->update_coordi(2, 6);
  x = pin0->get_center_coordi().get_x();
  y = pin0->get_center_coordi().get_y();
  EXPECT_EQ(7, x);
  EXPECT_EQ(9, y);
  x = pin1->get_center_coordi().get_x();
  y = pin1->get_center_coordi().get_y();
  EXPECT_EQ(8, x);
  EXPECT_EQ(7, y);

  delete inst;
  delete pin0;
  delete pin1;
}

}  // namespace ipl