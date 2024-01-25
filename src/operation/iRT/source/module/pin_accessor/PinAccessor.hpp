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
  PAModel initPAModel(std::vector<Net>& net_list);
  std::vector<PANet> convertToPANetList(std::vector<Net>& net_list);
  PANet convertToPANet(Net& net);
  void accessPANetList(PAModel& pa_model);
  void makeAccessPointList(PAModel& pa_model, PANet& pa_net);
  std::vector<LayerRect> getLegalShapeList(PAModel& pa_model, irt_int pa_net_idx, PAPin& pa_pin);
  std::vector<PlanarRect> getPlanarLegalRectList(PAModel& pa_model, irt_int pa_net_idx, std::vector<EXTLayerRect>& pin_shape_list);
  std::vector<AccessPoint> getAccessPointListByPrefTrackGrid(std::vector<LayerRect>& legal_shape_list);
  std::vector<AccessPoint> getAccessPointListByCurrTrackGrid(std::vector<LayerRect>& legal_shape_list);
  std::vector<AccessPoint> getAccessPointListByOnTrack(std::vector<LayerRect>& legal_shape_list);
  std::vector<AccessPoint> getAccessPointListByOnShape(std::vector<LayerRect>& legal_shape_list);
  void mergeAccessPointList(PANet& pa_net);
  void updateBoundingBox(PANet& pa_net);
  void updateAccessGrid(PANet& pa_net);
#if 1  // update
  void update(PAModel& pa_model);
#endif

#if 1  // plot pa_model
  void plotPAModel(PAModel& pa_model, std::string flag);
#endif
};

}  // namespace irt
