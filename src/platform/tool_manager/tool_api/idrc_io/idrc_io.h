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
#include <map>
#include <string>
#include <vector>

#include "ids.hpp"

namespace iplf {

#define drcInst DrcIO::getInstance()
class DrcIO
{
 public:
  static DrcIO* getInstance()
  {
    if (!_instance) {
      _instance = new DrcIO;
    }
    return _instance;
  }

  /// io
  bool runDRC(std::string config = "", std::string report_path = "", bool has_init = false);
  bool readDrcFromFile(std::string path = "");
  bool saveDrcToFile(std::string path);

  std::map<std::string, std::map<std::string, std::vector<ids::Violation>>>& getDetailCheckResult(std::string path = "");

 private:
  static DrcIO* _instance;

  DrcIO() {}
  ~DrcIO() = default;
};

}  // namespace iplf
