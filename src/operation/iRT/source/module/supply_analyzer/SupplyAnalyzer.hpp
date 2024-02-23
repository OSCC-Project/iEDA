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
// MERCHANTABILITY OR FIT FOR A SARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#pragma once

#include "Config.hpp"
#include "DataManager.hpp"
#include "Database.hpp"
#include "Net.hpp"
#include "SAModel.hpp"

namespace irt {

#define SA_INST (irt::SupplyAnalyzer::getInst())

class SupplyAnalyzer
{
 public:
  static void initInst();
  static SupplyAnalyzer& getInst();
  static void destroyInst();
  // function
  void analyze(std::vector<Net>& net_list);

 private:
  // self
  static SupplyAnalyzer* _sa_instance;

  SupplyAnalyzer() = default;
  SupplyAnalyzer(const SupplyAnalyzer& other) = delete;
  SupplyAnalyzer(SupplyAnalyzer&& other) = delete;
  ~SupplyAnalyzer() = default;
  SupplyAnalyzer& operator=(const SupplyAnalyzer& other) = delete;
  SupplyAnalyzer& operator=(SupplyAnalyzer&& other) = delete;
  // function
  SAModel initSAModel(std::vector<Net>& net_list);
  void buildLayerNodeMap(SAModel& sa_model);
  void buildSupplySchedule(SAModel& sa_model);
  void analyzeSupply(SAModel& sa_model);
  std::vector<LayerRect> getCrossingWireList(int32_t layer_idx, SANode& first_node, SANode& second_node);
  bool isAccess(LayerRect& wire, SANode& first_node, SANode& second_node);
  void updateSAModel(SAModel& sa_model);

#if 1  // exhibit
  void plotSAModel(SAModel& sa_model);
  void reportSAModel(SAModel& sa_model);
  void reportSummary(SAModel& sa_model);
  void writeSupplyCSV(SAModel& sa_model);
#endif
};

}  // namespace irt
