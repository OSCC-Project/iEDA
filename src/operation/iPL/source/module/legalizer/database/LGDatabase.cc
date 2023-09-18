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
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-02-08 11:03:27
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-20 11:09:32
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/LGDatabase.cc
 * @Description:
 *
 *
 */
#include "LGDatabase.hh"

namespace ipl {

LGDatabase::LGDatabase()
{
}

LGDatabase::~LGDatabase()
{
  delete _lg_layout;

  for (auto* inst : _lgInstance_list) {
    delete inst;
  }
  _lgInstance_list.clear();
  _lgInstance_map.clear();
  _instance_map.clear();
}

}  // namespace ipl