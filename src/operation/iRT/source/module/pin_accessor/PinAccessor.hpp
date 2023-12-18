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
#include "DRCShape.hpp"
#include "DataManager.hpp"
#include "Database.hpp"
#include "Net.hpp"
#include "PAModel.hpp"
#include "PANode.hpp"
#include "ViolatedGroup.hpp"

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
  void updateBlockageMap(PAModel& pa_model);
  void updateNetShapeMap(PAModel& pa_model);
  void checkPAModel(PAModel& pa_model);
#endif

#if 1  // iterative
  void iterative(PAModel& pa_model);
  void accessPAModel(PAModel& pa_model);
  void accessPANetList(PAModel& pa_model);
  void initAccessPointList(PAModel& pa_model, PANet& pa_net);
  std::vector<LayerRect> getLegalPinShapeList(PAModel& pa_model, irt_int pa_net_idx, PAPin& pa_pin);
  std::vector<PlanarRect> getViaLegalRectList(PAModel& pa_model, irt_int pa_net_idx, irt_int via_below_layer_idx,
                                              std::vector<EXTLayerRect>& pin_shape_list);
  std::vector<PlanarRect> getWireLegalRectList(PAModel& pa_model, irt_int pa_net_idx, std::vector<EXTLayerRect>& pin_shape_list);
  void mergeAccessPointList(PANet& pa_net);
  void updateAccessGridCoord(PANet& pa_net);
  void updateBoundingBox(PANet& pa_net);
  void updateAccessGrid(PANet& pa_net);
  void addViaAccessByDRC(PAModel& pa_model, PANet& pa_net);
  void addWireAccessByDRC(PAModel& pa_model, PANet& pa_net);
  LayerRect getOrientationWire(PlanarRect& base_region, LayerCoord& real_coord, Orientation orientation);
  void addWeakAccessByDRC(PAModel& pa_model, PANet& pa_net);
  void initProtectedPointList(PANet& pa_net);
  void initProtectedPointList(PAPin& pa_pin);
  void checkProtectedPointList(PANet& pa_net);
  void checkProtectedPointList(PAPin& pa_pin);
  void filterPANetList(PAModel& pa_model);
  void filterByAccessPointType(PAModel& pa_model);
  void filterByAccessPointType(PAPin& pa_pin);
  void filterByViaConflict(PAModel& pa_model);
  void updateNetCandidateViaMap(PAModel& pa_model);
  void filterByViaConflict(PANet& pa_net, PAModel& pa_model);
  void filterByAccessOrienSet(PAModel& pa_model);
  void filterByAccessOrienSet(PAPin& pa_pin);
  void filterByPinBalance(PAModel& pa_model);
  void makeProtectedAccessPoint(PAModel& pa_model);
  void makeProtectedAccessPoint(PAPin& pa_pin);
  void updateNetAccessPointMap(PAModel& pa_model);
  void optPANetList(PAModel& pa_model);
  std::vector<ViolatedGroup> getViolatedGroupList(PAModel& pa_model);
  void makeProtectedAccessPointList(PAModel& pa_model, std::vector<ViolatedGroup>& violated_group_list);
  void optProtectedAccessPoint(PAModel& pa_model, std::vector<ViolatedGroup>& violated_group_list);
  void countPAModel(PAModel& pa_model);
  void reportPAModel(PAModel& pa_model);
  bool stopPAModel(PAModel& pa_model);
#endif

#if 1  // update
  void update(PAModel& pa_model);
#endif

#if 1  // update env
  std::vector<DRCShape> getDRCShapeList(irt_int pa_net_idx, irt_int pa_pin_idx, std::vector<Segment<LayerCoord>>& segment_list);
  void updateAccessPointToUnit(PAModel& pa_model, ChangeType change_type, PASourceType pa_source_type, irt_int pa_net_idx,
                               irt_int pa_pin_idx, AccessPoint& access_point);
  void updateRectToUnit(PAModel& pa_model, ChangeType change_type, PASourceType pa_source_type, DRCShape drc_shape);
#endif

#if 1  // valid drc
  bool hasPAEnvViolation(PAModel& pa_model, PASourceType pa_source_type, const std::vector<DRCCheckType>& check_type_list,
                         const DRCShape& drc_shape);
  bool hasPAEnvViolation(PAModel& pa_model, PASourceType pa_source_type, const std::vector<DRCCheckType>& check_type_list,
                         const std::vector<DRCShape>& drc_shape_list);
  std::map<std::string, std::vector<ViolationInfo>> getPAEnvViolation(PAModel& pa_model, PASourceType pa_source_type,
                                                                      const std::vector<DRCCheckType>& check_type_list,
                                                                      const DRCShape& drc_shape);
  std::map<std::string, std::vector<ViolationInfo>> getPAEnvViolation(PAModel& pa_model, PASourceType pa_source_type,
                                                                      const std::vector<DRCCheckType>& check_type_list,
                                                                      const std::vector<DRCShape>& drc_shape_list);
  std::map<std::string, std::vector<ViolationInfo>> getPAEnvViolationBySingle(PAGCell& pa_gcell, PASourceType pa_source_type,
                                                                              const std::vector<DRCCheckType>& check_type_list,
                                                                              const std::vector<DRCShape>& drc_shape_list);
  void removeInvalidPAEnvViolationBySingle(PAGCell& pa_gcell, std::map<std::string, std::vector<ViolationInfo>>& drc_violation_map);
#endif
};

}  // namespace irt
