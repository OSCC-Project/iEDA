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
#include "Monitor.hpp"
#include "PPModel.hpp"

namespace irt {

#define RTPP (irt::PinPatcher::getInst())

class PinPatcher
{
 public:
  static void initInst();
  static PinPatcher& getInst();
  static void destroyInst();
  // function
  void patch();

 private:
  // self
  static PinPatcher* _pp_instance;

  PinPatcher() = default;
  PinPatcher(const PinPatcher& other) = delete;
  PinPatcher(PinPatcher&& other) = delete;
  ~PinPatcher() = default;
  PinPatcher& operator=(const PinPatcher& other) = delete;
  PinPatcher& operator=(PinPatcher&& other) = delete;
  // function
  PPModel initPPModel();
  std::vector<PPNet> convertToPPNetList(std::vector<Net>& net_list);
  PPNet convertToPPNet(Net& net);
  void uploadViolation(PPModel& pp_model);
  std::vector<Violation> getViolationList(PPModel& pp_model);
  void iterativePPModel(PPModel& pp_model);
  void setPPIterParam(PPModel& pp_model, int32_t iter, PPIterParam& pp_iter_param);
  void initPPBoxMap(PPModel& pp_model);
  void buildBoxSchedule(PPModel& pp_model);
  void routePPBoxMap(PPModel& pp_model);
  void buildFixedRect(PPBox& pp_box);
  void buildAccessResult(PPBox& pp_box);
  void buildViolation(PPBox& pp_box);
  void initNetIdxSet(PPBox& pp_box);
  void buildNetPatch(PPBox& pp_box);
  bool needRouting(PPBox& pp_box);
  void buildGraphShapeMap(PPBox& pp_box);
  void routePPBox(PPBox& pp_box);
  void initSingleTask(PPBox& pp_box);
  std::vector<PPSolution> getSolution(PPBox& pp_box);
  PPSolution getNewSolution(PPBox& pp_box);
  void updateCurrViolationList(PPBox& pp_box, PPSolution& pp_solution);
  std::vector<Violation> getViolationList(PPBox& pp_box);
  void updateCurrSolvedStatus(PPBox& pp_box);
  void updateTaskPatch(PPBox& pp_box);
  void updateViolationList(PPBox& pp_box);
  void resetSingleTask(PPBox& pp_box);
  void uploadNetPatch(PPBox& pp_box);
  void uploadViolation(PPBox& pp_box);
  void freePPBox(PPBox& pp_box);
  int32_t getViolationNum(PPModel& pp_model);
  void uploadNetPatch(PPModel& pp_model);
  bool stopIteration(PPModel& pp_model);

#if 1  // get env
  double getEnvCost(PPBox& pp_box, int32_t net_idx, EXTLayerRect& patch);
#endif

#if 1  // update env
  void updateFixedRectToGraph(PPBox& pp_box, ChangeType change_type, int32_t net_idx, EXTLayerRect* fixed_rect, bool is_routing);
  void updateFixedRectToGraph(PPBox& pp_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment);
  void updateRoutedRectToGraph(PPBox& pp_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment);
  void updateRoutedRectToGraph(PPBox& pp_box, ChangeType change_type, int32_t net_idx, EXTLayerRect& patch);
  void addViolationToGraph(PPBox& pp_box, Violation& violation);
  std::vector<PlanarRect> getGraphShape(PPBox& pp_box, NetShape& net_shape);
  std::vector<PlanarRect> getRoutingGraphShapeList(PPBox& pp_box, NetShape& net_shape);
#endif

#if 1  // exhibit
  void updateSummary(PPModel& pp_model);
  void printSummary(PPModel& pp_model);
  void outputNetCSV(PPModel& pp_model);
  void outputViolationCSV(PPModel& pp_model);
#endif

#if 1  // debug
  void debugPlotPPModel(PPModel& pp_model, std::string flag);
  void debugCheckPPBox(PPBox& pp_box);
  void debugPlotPPBox(PPBox& pp_box, std::string flag);
#endif
};

}  // namespace irt
