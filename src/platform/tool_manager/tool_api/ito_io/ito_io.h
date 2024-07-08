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
#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace iplf {

#define iTOInst (ToIO::getInstance())
class ToIO
{
 public:
  static ToIO* getInstance()
  {
    if (!_instance) {
      _instance = new ToIO;
    }
    return _instance;
  }

  /// io
  bool runTO(std::string config = "");
  bool runTOFixFanout(std::string config = "");
  bool runTODrv(std::string config = "");
  bool runTODrvSpecialNet(std::string config = "", std::string net_name = "");
  bool runTOHold(std::string config = "");
  bool runTOSetup(std::string config = "");
  bool runTOBuffering(std::string config = "", std::string net_name = "");

 private:
  static ToIO* _instance;

  ToIO() {}
  ~ToIO() = default;

  void resetConfig();
};

}  // namespace iplf
