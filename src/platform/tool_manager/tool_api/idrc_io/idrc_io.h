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

#include "file_drc.h"
#include "idrc_violation.h"

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

  /// getter
  int32_t get_buffer_size();
  std::map<std::string, std::vector<idrc::DrcViolation*>>& get_detail_drc() { return _detail_drc; }
  void set_detail_drc(std::map<std::string, std::vector<idrc::DrcViolation*>>& detail_drc);
  void clear();

  /// io
  bool runDRC(std::string config = "", std::string report_path = "");
  std::tuple<bool, std::vector<std::string>, std::vector<std::string>, int> checkConnnectivity();
  bool readDrcFromFile(std::string path = "");
  bool saveDrcToFile(std::string path);

  std::map<std::string, std::vector<idrc::DrcViolation*>> getDetailCheckResult(std::string path = "");

 private:
  static DrcIO* _instance;
  std::map<std::string, std::vector<idrc::DrcViolation*>> _detail_drc;

  DrcIO() {}
  ~DrcIO() = default;

  void get_def_drc();
};

}  // namespace iplf
