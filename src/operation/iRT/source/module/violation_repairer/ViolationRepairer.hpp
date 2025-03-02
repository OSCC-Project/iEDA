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
#include "VRModel.hpp"

namespace irt {

#define RTVR (irt::ViolationRepairer::getInst())

class ViolationRepairer
{
 public:
  static void initInst();
  static ViolationRepairer& getInst();
  static void destroyInst();
  // function
  void repair();

 private:
  // self
  static ViolationRepairer* _vr_instance;

  ViolationRepairer() = default;
  ViolationRepairer(const ViolationRepairer& other) = delete;
  ViolationRepairer(ViolationRepairer&& other) = delete;
  ~ViolationRepairer() = default;
  ViolationRepairer& operator=(const ViolationRepairer& other) = delete;
  ViolationRepairer& operator=(ViolationRepairer&& other) = delete;
  // function
  VRModel initVRModel();
  std::vector<VRNet> convertToVRNetList(std::vector<Net>& net_list);
  VRNet convertToVRNet(Net& net);
  void updateAccessPoint(VRModel& vr_model);
  void initNetFinalResultMap(VRModel& vr_model);
  void buildNetFinalResultMap(VRModel& vr_model);
  void clearIgnoredViolation(VRModel& vr_model);
  void uploadViolation(VRModel& vr_model);
  std::vector<Violation> getHybridNetViolationList(VRModel& vr_model);
  void iterativeVRModel(VRModel& vr_model);
  void setVRIterParam(VRModel& vr_model, int32_t iter, VRIterParam& vr_iter_param);
  void initVRBoxMap(VRModel& vr_model);
  void buildBoxSchedule(VRModel& vr_model);
  void splitNetResult(VRModel& vr_model);
  void routeVRBoxMap(VRModel& vr_model);
  void buildFixedRect(VRBox& vr_box);
  void buildViolation(VRBox& vr_box);
  void initNetTaskIdxSet(VRBox& vr_box);
  void buildNetResult(VRBox& vr_box);
  void buildNetPatch(VRBox& vr_box);
  bool needRouting(VRBox& vr_box);
  void buildBoxTrackAxis(VRBox& vr_box);
  void buildGraphShapeMap(VRBox& vr_box);
  void routeVRBox(VRBox& vr_box);
  void initSingleTask(VRBox& vr_box, ViolationType& violation_type);
  std::vector<VRSolution> getVRSolutionList(VRBox& vr_box, ViolationType& violation_type);
  std::vector<VRSolution> routeByCutShort(VRBox& vr_box);
  std::vector<VRSolution> routeBySameLayerCutSpacing(VRBox& vr_box);
  std::vector<VRSolution> routeByParallelRunLengthSpacing(VRBox& vr_box);
  std::vector<VRSolution> routeByMinimumArea(VRBox& vr_box);
  VRSolution getNewSolution(VRBox& vr_box);
  void updateCurrResultList(VRBox& vr_box, VRSolution& vr_solution);
  void updateCurrViolationList(VRBox& vr_box);
  std::vector<Violation> getHybridNetViolationList(VRBox& vr_box);
  void updateCurrSolvedStatus(VRBox& vr_box);
  void updateTaskResult(VRBox& vr_box);
  void updateTaskPatch(VRBox& vr_box);
  void updateViolationList(VRBox& vr_box);
  void resetSingleTask(VRBox& vr_box);
  void uploadNetResult(VRBox& vr_box);
  void uploadNetPatch(VRBox& vr_box);
  void uploadViolation(VRBox& vr_box);
  void freeVRBox(VRBox& vr_box);
  int32_t getViolationNum(VRModel& vr_model);
  void uploadNetResult(VRModel& vr_model);
  void uploadNetPatch(VRModel& vr_model);
  bool stopIteration(VRModel& vr_model);

#if 1  // get env
  double getEnvCost(VRBox& vr_box, int32_t net_idx, Segment<LayerCoord>& segment);
  double getEnvCost(VRBox& vr_box, int32_t net_idx, EXTLayerRect& patch);
#endif

#if 1  // update env
  void updateFixedRectToGraph(VRBox& vr_box, ChangeType change_type, int32_t net_idx, EXTLayerRect* fixed_rect, bool is_routing);
  void updateFixedRectToGraph(VRBox& vr_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment);
  void updateRoutedRectToGraph(VRBox& vr_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment);
  void updateRoutedRectToGraph(VRBox& vr_box, ChangeType change_type, int32_t net_idx, EXTLayerRect& patch);
  void addViolationToGraph(VRBox& vr_box, Violation& violation);
  std::vector<PlanarRect> getGraphShape(VRBox& vr_box, NetShape& net_shape);
  std::vector<PlanarRect> getRoutingGraphShapeList(VRBox& vr_box, NetShape& net_shape);
  std::vector<PlanarRect> getCutGraphShapeList(VRBox& vr_box, NetShape& net_shape);
#endif

#if 1  // exhibit
  void updateSummary(VRModel& vr_model);
  void printSummary(VRModel& vr_model);
  void outputNetCSV(VRModel& vr_model);
  void outputViolationCSV(VRModel& vr_model);
#endif

#if 1  // debug
  void debugPlotVRModel(VRModel& vr_model, std::string flag);
  void debugCheckVRBox(VRBox& vr_box);
  void debugPlotVRBox(VRBox& vr_box, std::string flag);
#endif
};

}  // namespace irt
