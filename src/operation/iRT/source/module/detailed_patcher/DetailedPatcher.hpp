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
#include "DPModel.hpp"
#include "DataManager.hpp"
#include "Database.hpp"
#include "Monitor.hpp"

namespace irt {

#define RTDP (irt::DetailedPatcher::getInst())

class DetailedPatcher
{
 public:
  static void initInst();
  static DetailedPatcher& getInst();
  static void destroyInst();
  // function
  void patch();

 private:
  // self
  static DetailedPatcher* _dp_instance;

  DetailedPatcher() = default;
  DetailedPatcher(const DetailedPatcher& other) = delete;
  DetailedPatcher(DetailedPatcher&& other) = delete;
  ~DetailedPatcher() = default;
  DetailedPatcher& operator=(const DetailedPatcher& other) = delete;
  DetailedPatcher& operator=(DetailedPatcher&& other) = delete;
  // function
  DPModel initDPModel();
  std::vector<DPNet> convertToDPNetList(std::vector<Net>& net_list);
  DPNet convertToDPNet(Net& net);
  void uploadViolation(DPModel& dp_model);
  std::vector<Violation> getViolationList(DPModel& dp_model);
  void iterativeDPModel(DPModel& dp_model);
  void setDPIterParam(DPModel& dp_model, int32_t iter, DPIterParam& dp_iter_param);
  void initDPBoxMap(DPModel& dp_model);
  void buildBoxSchedule(DPModel& dp_model);
  void routeDPBoxMap(DPModel& dp_model);
  void buildFixedRect(DPBox& dp_box);
  void buildNetResult(DPBox& dp_box);
  void buildNetPatch(DPBox& dp_box);
  void buildViolation(DPBox& dp_box);
  bool needRouting(DPBox& dp_box);
  bool isValid(DPBox& dp_box, Violation& violation);
  void buildGraphShapeMap(DPBox& dp_box);
  void routeDPBox(DPBox& dp_box);
  void initSingleTask(DPBox& dp_box);
  std::vector<DPSolution> getSolution(DPBox& dp_box);
  DPSolution getNewSolution(DPBox& dp_box);
  void updateCurrViolationList(DPBox& dp_box, DPSolution& dp_solution);
  std::vector<Violation> getViolationList(DPBox& dp_box);
  void updateCurrSolvedStatus(DPBox& dp_box);
  void updateTaskPatch(DPBox& dp_box);
  void updateViolationList(DPBox& dp_box);
  void resetSingleTask(DPBox& dp_box);
  void uploadNetPatch(DPBox& dp_box);
  void freeDPBox(DPBox& dp_box);
  int32_t getViolationNum(DPModel& dp_model);
  void uploadNetPatch(DPModel& dp_model);
  bool stopIteration(DPModel& dp_model);

#if 1  // get env
  double getEnvCost(DPBox& dp_box, int32_t net_idx, EXTLayerRect& patch);
#endif

#if 1  // update env
  void updateFixedRectToGraph(DPBox& dp_box, ChangeType change_type, int32_t net_idx, EXTLayerRect* fixed_rect, bool is_routing);
  void updateFixedRectToGraph(DPBox& dp_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment);
  void updateRoutedRectToGraph(DPBox& dp_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment);
  void updateRoutedRectToGraph(DPBox& dp_box, ChangeType change_type, int32_t net_idx, EXTLayerRect& patch);
  std::vector<PlanarRect> getGraphShape(DPBox& dp_box, NetShape& net_shape);
  std::vector<PlanarRect> getRoutingGraphShapeList(DPBox& dp_box, NetShape& net_shape);
#endif

#if 1  // exhibit
  void updateSummary(DPModel& dp_model);
  void printSummary(DPModel& dp_model);
  void outputNetCSV(DPModel& dp_model);
  void outputViolationCSV(DPModel& dp_model);
#endif

#if 1  // debug
  void debugPlotDPModel(DPModel& dp_model, std::string flag);
  void debugCheckDPBox(DPBox& dp_box);
  void debugPlotDPBox(DPBox& dp_box, std::string flag);
#endif
};

}  // namespace irt
