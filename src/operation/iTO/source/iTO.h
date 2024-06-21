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

#include "ids.hpp"

namespace ito {

class DbInterface;
class ToConfig;
class Tree;

class iTO {
 public:
  iTO() = delete;
  iTO(const std::string &config_file);
  ~iTO();
  iTO(const iTO &other) = delete;
  iTO(iTO &&other) = delete;

  DbInterface *get_db_interface() {
    // initialization();
    return _db_interface;
  }

  ToConfig *get_config() { return _to_config; }

  /// operator
  void initialization(idb::IdbBuilder *idb_build, ista::TimingEngine *timing);
  void resetInitialization(idb::IdbBuilder    *idb_build,
                           ista::TimingEngine *timing_engine = nullptr);

  void runTO();
  void optimizeDesignViolation();
  void optimizeSetup();
  void optimizeHold();


 private:
  DbInterface *_db_interface = nullptr;
  ToConfig    *_to_config = nullptr;
};

} // namespace ito
