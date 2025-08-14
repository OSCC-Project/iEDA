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
 * @file PNPIdbWrapper.hh
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "PNPGridManager.hh"

namespace ipnp {

class PNPIdbWrapper
{
 public:
  PNPIdbWrapper() = default;
  ~PNPIdbWrapper() = default;

  void saveToIdb(PNPGridManager pnp_network);
  void writeIdbToDef(std::string def_file_path);

  void connect_M2_M1_Layer();

 private:
};

}  // namespace ipnp
