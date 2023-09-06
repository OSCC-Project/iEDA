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
#include "DRCChecker.hpp"
#include "DRCRect.hpp"
#include "DataManager.hpp"
#include "Database.hpp"
#include "Net.hpp"
#include "PAModel.hpp"
#include "PANode.hpp"

namespace irt {

#define PA_INST (irt::PinAccessor::getInst())

class PinAccessor
{
 public:
  static void initInst();
  static PinAccessor& getInst();
  static void destroyInst();
  // function
  void access(std::vector<Net>& net_list);

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
  void accessNetList(std::vector<Net>& net_list);

#if 1  // init
  PAModel init(std::vector<Net>& net_list);
  PAModel initPAModel(std::vector<Net>& net_list);
  std::vector<PANet> convertToPANetList(std::vector<Net>& net_list);
  PANet convertToPANet(Net& net);
  void buildPAModel(PAModel& pa_model);
  void updateNetFixedRectMap(PAModel& pa_model);
  void addRectToEnv(PAModel& pa_model, PASourceType pa_source_type, DRCRect drc_rect);
  void checkPAModel(PAModel& pa_model);
#endif

#if 1  // iterative
  void iterative(PAModel& pa_model);
  void accessPAModel(PAModel& pa_model);
  void accessPANetList(PAModel& pa_model);
  void accessPANet(PAModel& pa_model, PANet& pa_net);
  void initAccessPointList(PAModel& pa_model, PANet& pa_net);
  std::vector<LayerRect> getLegalPinShapeList(PAModel& pa_model, irt_int pa_net_idx, PAPin& pa_pin);
  std::vector<PlanarRect> getViaLegalRectList(PAModel& pa_model, irt_int pa_net_idx, irt_int via_below_layer_idx,
                                              std::vector<EXTLayerRect>& pin_shape_list);
  void mergeLegalRectList(std::vector<LayerRect>& legal_rect_list);
  std::vector<PlanarRect> getWireLegalRectList(PAModel& pa_model, irt_int pa_net_idx, std::vector<EXTLayerRect>& pin_shape_list);
  void mergeAccessPointList(PANet& pa_net);
  void selectAccessPointByType(PANet& pa_net);
  void updateAccessGridCoord(PANet& pa_net);
  void updateBoundingBox(PANet& pa_net);
  void updateAccessGrid(PANet& pa_net);
  void selectAccessPointByGCell(PANet& pa_net);
  void eliminateDRCViolation(PAModel& pa_model, PANet& pa_net);
  void checkAccessPointList(PANet& pa_net);
  void updateNetCandidateViaMap(PAModel& pa_model);
  void eliminateViaConflict(PAModel& pa_model);
  void selectByViaNumber(PANet& pa_net, PAModel& pa_model);
  void selectByNetDistance(PANet& pa_net);
  void checkAccessPointNum(PANet& pa_net);
  void processPAModel(PAModel& pa_model);
  void updateAccessPointList(PAModel& pa_model);
  void updateDrivingPin(PANet& pa_net);
  void updateNetAccessPointMap(PAModel& pa_model);
  void countPAModel(PAModel& pa_model);
  void reportPAModel(PAModel& pa_model);
  bool stopPAModel(PAModel& pa_model);
#endif

#if 1  // update
  void update(PAModel& pa_model);
#endif

#if 1  // valid drc
  bool hasViolation(PAModel& pa_model, PASourceType pa_source_type, const std::vector<DRCRect>& drc_rect_list);
  std::map<std::string, std::vector<ViolationInfo>> getViolationInfo(PAGCell& pa_gcell, PASourceType pa_source_type,
                                                                     const std::vector<DRCRect>& drc_rect_list);
  std::map<std::string, std::vector<ViolationInfo>> getViolationInfo(PAGCell& pa_gcell, PASourceType pa_source_type);
  void removeInvalidViolationInfo(PAGCell& pa_gcell, std::map<std::string, std::vector<ViolationInfo>>& drc_violation_map);
#endif
};

}  // namespace irt
