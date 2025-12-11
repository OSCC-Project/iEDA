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

#include <any>
#include <map>
#include <string>
#include <vector>

#include "ids.hpp"
#include "../source/io/DbInterface.h"
#include "../iNO.h"

namespace ieda_feature {
  struct NetOptSummary;
}// namespace

namespace ino {

#define NoApiInst (ino::NoApi::getInst())

class NoApi {
 public:
  static NoApi &getInst();
  static void   destroyInst();

  void initNO(const std::string &ITO_CONFIG_PATH);
  void iNODataInit(idb::IdbBuilder *idb = nullptr, ista::TimingEngine *timing = nullptr);
  // void resetiTOData(idb::IdbBuilder *idb, ista::TimingEngine *timing = nullptr);
  // // function API
  void fixFanout();
  void fixIO();

  void saveDef(std::string saved_def_path = "");

  NoConfig *get_no_config();

  void reportTiming();

  std::vector<EvalData> getEvalData() { return _ino->get_db_interface()->eval_data(); }
  ieda_feature::NetOptSummary outputSummary();

 private:
  static NoApi *_no_api_instance;
  NoApi() = default;
  NoApi(const NoApi &other) = delete;
  NoApi(NoApi &&other) = delete;
  ~NoApi() = default;
  NoApi &operator=(const NoApi &other) = delete;
  NoApi &operator=(NoApi &&other) = delete;

  idb::IdbBuilder    *initIDB();
  ista::TimingEngine *initISTA(idb::IdbBuilder *idb);

  ino::iNO           *_ino = nullptr;
  idb::IdbBuilder    *_idb = nullptr;
  ista::TimingEngine *_timing_engine = nullptr;
};

} // namespace ino
