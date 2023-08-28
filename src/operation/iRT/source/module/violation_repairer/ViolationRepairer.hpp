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
#include "DRCRect.hpp"
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
  void updateNetFixedRectMap(VRModel& vr_model);
  void addRectToEnv(VRModel& vr_model, VRSourceType vr_source_type, DRCRect drc_rect);
  void updateVRResultTree(VRModel& vr_model);
  void buildKeyCoordPinMap(VRNet& vr_net);
  void buildCoordTree(VRNet& vr_net);
  void buildPHYNodeResult(VRNet& vr_net);
  void updateConnectionList(TNode<LayerCoord>* coord_node, VRNet& vr_net, std::vector<TNode<PHYNode>*>& pre_connection_list,
                            std::vector<TNode<PHYNode>*>& post_connection_list);
  TNode<PHYNode>* makeWirePHYNode(VRNet& vr_net, LayerCoord first_coord, LayerCoord second_coord);
  TNode<PHYNode>* makeViaPHYNode(VRNet& vr_net, irt_int below_layer_idx, PlanarCoord coord);
  TNode<PHYNode>* makePinPHYNode(VRNet& vr_net, irt_int pin_idx, LayerCoord coord);
  void updateNetResultMap(VRModel& vr_model, VRNet& vr_net);
  void checkVRModel(VRModel& vr_model);
#endif

#if 1  // iterative
  void iterative(VRModel& vr_model);
  void repairVRModel(VRModel& vr_model);
  void repairAntenna(VRModel& vr_model);
  void repairMinStep(VRModel& vr_model);
  void repairMinArea(VRModel& vr_model);
  void repairMinArea(VRModel& vr_model, VRNet& vr_net);
  bool hasViolation(VRModel& vr_model, VRSourceType vr_source_type, const std::vector<DRCRect>& drc_rect_list);
  void countVRModel(VRModel& vr_model);
  void reportVRModel(VRModel& vr_model);
  bool stopVRModel(VRModel& vr_model);
#endif

#if 1  // update
  void update(VRModel& vr_model);
#endif
};

}  // namespace irt
