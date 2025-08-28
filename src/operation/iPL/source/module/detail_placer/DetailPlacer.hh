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
/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-03-01 17:06:44
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-10 11:03:00
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/DetailPlacer.hh
 * @Description: Main for detail placement
 *
 *
 */
#ifndef IPL_DETAILPLACER_H
#define IPL_DETAILPLACER_H

#include "DPOperator.hh"
#include "GridManager.hh"
#include "PlacerDB.hh"
#include "TopologyManager.hh"
#include "database/DPDatabase.hh"
#include "AIWirelength.hh"

namespace ipl {

class DetailPlacer
{
 public:
  DetailPlacer() = delete;
  DetailPlacer(Config* pl_config, PlacerDB* placer_db);
  DetailPlacer(const DetailPlacer&) = delete;
  DetailPlacer(DetailPlacer&&) = delete;
  ~DetailPlacer();

  DetailPlacer& operator=(const DetailPlacer&) = delete;
  DetailPlacer& operator=(DetailPlacer&&) = delete;

  bool checkIsLegal();
  void runDetailPlace();
  int64_t calTotalHPWL();
  float calPeakBinDensity();

  void runDetailPlaceNFS();

  // AI wirelength prediction methods
  bool loadAIWirelengthModel(const std::string& model_path);
  bool loadAIWirelengthNormalizationParams(const std::string& params_path);
  void setUseAIWirelength(bool use_ai);
  int64_t calTotalAIWirelength();

 private:
  DPConfig _config;
  DPDatabase _database;
  DPOperator _operator;
  std::unique_ptr<AIWirelength> _ai_wirelength_evaluator;
  bool _use_ai_wirelength = false;

  void initDPConfig(Config* pl_config);
  void initDPDatabase(PlacerDB* placer_db);
  void initDPLayout();
  void wrapRowList();
  void wrapRegionList();
  void wrapCellList();
  void initDPDesign();
  void wrapInstanceList();
  DPInstance* wrapInstance(Instance* pl_inst);
  void wrapNetList();
  DPNet* wrapNet(Net* pl_net);
  DPPin* wrapPin(Pin* pl_pin);
  void updateInstanceList();
  void correctOutsidePinCoordi();
  void initIntervalList();

  void clearClusterInfo();
  void alignInstanceOrient();

  void notifyPLPlaceDensity();
};
}  // namespace ipl

#endif