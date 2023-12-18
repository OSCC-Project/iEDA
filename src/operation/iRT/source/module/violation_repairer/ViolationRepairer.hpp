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
#include "VRModel.hpp"

namespace irt {

#define VR_INST (irt::ViolationRepairer::getInst())

class ViolationRepairer
{
 public:
  static void initInst();
  static ViolationRepairer& getInst();
  static void destroyInst();
  // function
  void repair(std::vector<Net>& net_list);

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
  void repairNetList(std::vector<Net>& net_list);

#if 1  // init
  VRModel init(std::vector<Net>& net_list);
  VRModel initVRModel(std::vector<Net>& net_list);
  std::vector<VRNet> convertToVRNetList(std::vector<Net>& net_list);
  VRNet convertToVRNet(Net& net);
  void buildVRModel(VRModel& vr_model);
  void updateBlockageMap(VRModel& vr_model);
  void updateNetShapeMap(VRModel& vr_model);
  void calcVRGCellSupply(VRModel& vr_model);
  std::vector<PlanarRect> getWireList(VRGCell& vr_gcell, RoutingLayer& routing_layer);
  void updateVRResultTree(VRModel& vr_model);
  void buildKeyCoordPinMap(VRNet& vr_net);
  void buildCoordTree(VRNet& vr_net);
  void buildPhysicalNodeResult(VRNet& vr_net);
  void updateConnectionList(TNode<LayerCoord>* coord_node, VRNet& vr_net, std::vector<TNode<PhysicalNode>*>& pre_connection_list,
                            std::vector<TNode<PhysicalNode>*>& post_connection_list);
  TNode<PhysicalNode>* makeWirePhysicalNode(VRNet& vr_net, LayerCoord first_coord, LayerCoord second_coord);
  TNode<PhysicalNode>* makeViaPhysicalNode(VRNet& vr_net, irt_int below_layer_idx, PlanarCoord coord);
  TNode<PhysicalNode>* makePinPhysicalNode(VRNet& vr_net, irt_int pin_idx, LayerCoord coord);
  void checkVRModel(VRModel& vr_model);
#endif

#if 1  // iterative
  void iterative(VRModel& vr_model);
  void repairVRModel(VRModel& vr_model);
  void repairAntenna(VRModel& vr_model);
  void repairMinStep(VRModel& vr_model);
  void repairMinArea(VRModel& vr_model);
  void repairMinArea(VRModel& vr_model, VRNet& vr_net);
  void processVRModel(VRModel& vr_model);
  void updateNetResultMap(VRModel& vr_model);
  void calcVRGCellDemand(VRModel& vr_model);
  void countVRModel(VRModel& vr_model);
  void reportVRModel(VRModel& vr_model);
  bool stopVRModel(VRModel& vr_model);
#endif

#if 1  // update
  void update(VRModel& vr_model);
#endif

#if 1  // update env
  std::vector<DRCShape> getDRCShapeList(irt_int vr_net_idx, std::vector<Segment<LayerCoord>>& segment_list);
  std::vector<DRCShape> getDRCShapeList(irt_int vr_net_idx, MTree<PhysicalNode>& physical_node_tree);
  void updateRectToUnit(VRModel& vr_model, ChangeType change_type, VRSourceType vr_source_type, DRCShape drc_shape);
#endif

#if 1  // valid drc
  bool hasVREnvViolation(VRModel& vr_model, VRSourceType vr_source_type, const std::vector<DRCCheckType>& check_type_list,
                         const DRCShape& drc_shape);
  bool hasVREnvViolation(VRModel& vr_model, VRSourceType vr_source_type, const std::vector<DRCCheckType>& check_type_list,
                         const std::vector<DRCShape>& drc_shape_list);
  std::map<std::string, std::vector<ViolationInfo>> getVREnvViolation(VRModel& vr_model, VRSourceType vr_source_type,
                                                                      const std::vector<DRCCheckType>& check_type_list,
                                                                      const DRCShape& drc_shape);
  std::map<std::string, std::vector<ViolationInfo>> getVREnvViolation(VRModel& vr_model, VRSourceType vr_source_type,
                                                                      const std::vector<DRCCheckType>& check_type_list,
                                                                      const std::vector<DRCShape>& drc_shape_list);
  std::map<std::string, std::vector<ViolationInfo>> getVREnvViolationBySingle(VRGCell& vr_gcell, VRSourceType vr_source_type,
                                                                              const std::vector<DRCCheckType>& check_type_list,
                                                                              const std::vector<DRCShape>& drc_shape_list);
  void removeInvalidVREnvViolationBySingle(VRGCell& vr_gcell, std::map<std::string, std::vector<ViolationInfo>>& drc_violation_map);
  std::map<std::string, std::vector<ViolationInfo>> getVRSelfViolationInfo(const std::vector<DRCCheckType>& check_type_list,
                                                                           const std::vector<DRCShape>& drc_shape_list);
  void removeInvalidVRSelfViolationInfo(std::map<std::string, std::vector<ViolationInfo>>& drc_violation_map);
#endif
};

}  // namespace irt
