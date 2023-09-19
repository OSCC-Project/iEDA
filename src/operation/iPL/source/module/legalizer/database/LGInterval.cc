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
 * @Date: 2023-02-07 21:18:56
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-20 14:56:02
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/LGInterval.cc
 * @Description:
 *
 *
 */
#include "LGInterval.hh"

namespace ipl {

LGInterval::LGInterval(std::string name, int32_t min_x, int32_t max_x)
    : _index(-1), _name(name), _belong_row(nullptr), _min_x(min_x), _max_x(max_x)
{
}

LGInterval::~LGInterval()
{
}

void LGInterval::reset()
{
}

}  // namespace ipl