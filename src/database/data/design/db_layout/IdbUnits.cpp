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
/**
 * @project		iDB
 * @file		IdbUnits.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe lef Units information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbUnits.h"

namespace idb {

IdbUnits::IdbUnits()
{
  _nanoseconds = -1;
  _picofarads = -1;
  _ohms = -1;
  _milliwatts = -1;
  _milliamps = -1;
  _volts = -1;
  _micron_dbu = -1;
  _megahertz = -1;
}

void IdbUnits::print()
{
  std::cout << "nanoseconds = " << _nanoseconds << " picofarads = " << _picofarads << " ohms = "
            << " milliwatts = " << _milliwatts << " milliamps = " << _milliamps << " volts = " << _volts << " micron_dbu = " << _micron_dbu
            << " megahertz = " << _megahertz << std::endl;
}

}  // namespace idb
