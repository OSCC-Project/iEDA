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

#include "Config.hpp"
#include "DataManager.hpp"
#include "Database.hpp"
#include "Net.hpp"
#include "SAModel.hpp"

namespace irt {

#define RTSA (irt::SupplyAnalyzer::getInst())

class SupplyAnalyzer
{
 public:
  static void initInst();
  static SupplyAnalyzer& getInst();
  static void destroyInst();
  // function
  void analyze();

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
  SAModel initSAModel();
  void setSAComParam(SAModel& sa_model);
  void buildSupplySchedule(SAModel& sa_model);
  void analyzeSupply(SAModel& sa_model);
  EXTLayerRect getSearchRect(LayerCoord& first_coord, LayerCoord& second_coord);
  std::vector<LayerRect> getCrossingWireList(EXTLayerRect& search_rect);
  bool isAccess(LayerRect& wire, std::vector<PlanarRect>& obs_rect_list);
  void reduceSupply(SAModel& sa_model);
  void buildIgnoreNet(SAModel& sa_model);
  void analyzeDemandUnit(SAModel& sa_model);

#if 1  // exhibit
  void updateSummary(SAModel& sa_model);
  void printSummary(SAModel& sa_model);
  void outputPlanarSupplyCSV(SAModel& sa_model);
  void outputLayerSupplyCSV(SAModel& sa_model);
#endif

#if 1  // debug
  void debugPlotSAModel(SAModel& sa_model);
#endif
};

}  // namespace irt
