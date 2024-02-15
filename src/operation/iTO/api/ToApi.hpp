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
#include "../iTO.h"

namespace ito {

#define ToApiInst (ito::ToApi::getInst())

class ToApi {
 public:
  static ToApi &getInst();
  static void   destroyInst();

  void initTO(const std::string &ITO_CONFIG_PATH);
  void iTODataInit(idb::IdbBuilder *idb = nullptr, ista::TimingEngine *timing = nullptr);
  void resetiTOData(idb::IdbBuilder *idb, ista::TimingEngine *timing = nullptr);
  // function API
  void runTO();
  void optimizeDesignViolation();
  void optimizeSetup();
  void optimizeHold();

  void initCTSDesignViolation(idb::IdbBuilder *idb, ista::TimingEngine *timing);
  std::vector<idb::IdbNet *> optimizeCTSDesignViolation(idb::IdbNet *idb_net, Tree *topo);

  void saveDef(std::string saved_def_path = "");

  ToConfig *get_to_config();
  void resetConfigLibs(std::vector<std::string>& paths);
  void resetConfigSdc(std::string& path);
  Tree     *get_tree(const int&size);
  void addTopoEdge(Tree *topo, const int &first_id, const int &second_id, const int &x1,
                   const int &y1, const int &x2, const int &y2);
  void topoIdToDesignObject(ito::Tree *topo, const int &id, ista::DesignObject *sta_pin);
  void topoSetDriverId(ito::Tree *topo, const int &id);
  void reportTiming();
  std::vector<EvalData> getEvalData() { return _ito->get_db_interface()->eval_data(); }

 private:
  static ToApi *_to_api_instance;
  ToApi() = default;
  ToApi(const ToApi &other) = delete;
  ToApi(ToApi &&other) = delete;
  ~ToApi() = default;
  ToApi &operator=(const ToApi &other) = delete;
  ToApi &operator=(ToApi &&other) = delete;

  idb::IdbBuilder    *initIDB();
  ista::TimingEngine *initISTA(idb::IdbBuilder *idb);

  ito::iTO           *_ito = nullptr;
  idb::IdbBuilder    *_idb = nullptr;
  ista::TimingEngine *_timing_engine = nullptr;
};

} // namespace ito
