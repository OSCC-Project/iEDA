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
 * @Date: 2022-03-09 17:30:51
 * @LastEditTime: 2023-03-03 20:14:08
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @Description:
 * @FilePath: /irefactor/src/operation/iPL/source/module/evaluator/density/Density.cc
 * Contact : https://github.com/sjchanson
 */

#include "Density.hh"

namespace ipl {

int64_t Density::obtainOverflowArea()
{
  return _grid_manager->obtainTotalOverflowArea();
}

std::vector<Grid*> Density::obtainOverflowIllegalGridList()
{
  std::vector<Grid*> illegal_grid_list;
  _grid_manager->obtainOverflowIllegalGridList(illegal_grid_list);
  return illegal_grid_list;
}

float Density::obtainPeakBinDensity()
{
  return _grid_manager->obtainPeakGridDensity();
}

}  // namespace ipl