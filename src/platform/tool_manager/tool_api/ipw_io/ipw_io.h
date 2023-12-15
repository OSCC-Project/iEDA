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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#define powerInst iplf::PowerIO::getInstance()

namespace idb {
class IdbBuilder;
enum class IdbConnectType : uint8_t;
}  // namespace idb

namespace iplf {

class PowerIO
{
 public:
  static PowerIO* getInstance()
  {
    if (!_instance) {
      _instance = new PowerIO;
    }
    return _instance;
  }

  /// getter

  /// io
  bool autoRunPower(std::string path = "");
  bool runPower(std::string path = "");

 private:
  static PowerIO* _instance;

  PowerIO() {}
  ~PowerIO() = default;

  bool reportSummaryPower();
};

}  // namespace iplf
