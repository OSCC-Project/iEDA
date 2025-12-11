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

#include "FixFanout.h"

#include <iostream>
#include <string>

namespace ino {

class NoConfig;
class DbInterface;

class iNO {
 public:
  iNO() = delete;
  iNO(const std::string &config_file);
  ~iNO();

  DbInterface *get_db_interface() { return _db_interface; }
  NoConfig *get_config() { return _no_config; }

  void fixFanout();
  void fixIO();

  void initialization(idb::IdbBuilder *idb_build, ista::TimingEngine *timing);
 private:

  // data
  DbInterface *_db_interface;
  NoConfig    *_no_config = nullptr;
};

} // namespace ino
