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
#include "Net.hpp"
#include "PAModel.hpp"

namespace irt {

#define RTPA (irt::PinAccessor::getInst())

class PinAccessor
{
 public:
  static void initInst();
  static PinAccessor& getInst();
  static void destroyInst();
  // function
  void access();

 private:
  // self
  static PinAccessor* _pa_instance;

  PinAccessor() = default;
  PinAccessor(const PinAccessor& other) = delete;
  PinAccessor(PinAccessor&& other) = delete;
  ~PinAccessor() = default;
  PinAccessor& operator=(const PinAccessor& other) = delete;
  PinAccessor& operator=(PinAccessor&& other) = delete;
  // function
  PAModel initPAModel();
  std::vector<PANet> convertToPANetList(std::vector<Net>& net_list);
  PANet convertToPANet(Net& net);
  void initLayerEnclosureMap(PAModel& pa_model);
  void initAccessPointList(PAModel& pa_model);
  std::vector<LayerRect> getLegalShapeList(PAModel& pa_model, int32_t net_idx, Pin* pin);
  std::vector<PlanarRect> getPlanarLegalRectList(PAModel& pa_model, int32_t curr_net_idx, std::vector<EXTLayerRect>& pin_shape_list);
  std::vector<AccessPoint> getAccessPointListByTrackGrid(int32_t pin_idx, std::vector<LayerRect>& legal_shape_list);
  std::vector<AccessPoint> getAccessPointListByOnTrack(int32_t pin_idx, std::vector<LayerRect>& legal_shape_list);
  std::vector<AccessPoint> getAccessPointListByShapeCenter(int32_t pin_idx, std::vector<LayerRect>& legal_shape_list);
  void buildAccessPointList(PAModel& pa_model);
  void uploadAccessPoint(PAModel& pa_model);
  void buildConflictGroupList(PAModel& pa_model);
  std::map<PAPin*, std::set<PAPin*>> getPinConlictMap(PAModel& pa_model);
  bool hasConflict(PAModel& pa_model, AccessPoint& access_point_a, AccessPoint& access_point_b);
  void eliminateConflict(PAModel& pa_model);
  vector<ConflictAccessPoint> getBestPointList(const std::vector<vector<ConflictAccessPoint>>& conflict_ap_list_list);
  int32_t getMinDistance(std::vector<ConflictAccessPoint>& point_list);
  void updatePAModel(PAModel& pa_model);

#if 1  // exhibit
  void updateSummary(PAModel& pa_model);
  void printSummary(PAModel& pa_model);
  void writePlanarPinCSV(PAModel& pa_model);
  void writeLayerPinCSV(PAModel& pa_model);
#endif

#if 1  // debug
  void debugPlotPAModel(PAModel& pa_model, std::string flag);
#endif
};

}  // namespace irt
