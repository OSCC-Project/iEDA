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
  PAModel initPAModel(std::vector<Net>& net_list);
  std::vector<PANet> convertToPANetList(std::vector<Net>& net_list);
  PANet convertToPANet(Net& net);
  void initAccessPointList(PAModel& pa_model);
  std::vector<LayerRect> getLegalShapeList(PAModel& pa_model, int32_t pa_net_idx, PAPin* pa_pin);
  std::vector<PlanarRect> getPlanarLegalRectList(PAModel& pa_model, int32_t pa_net_idx, std::vector<EXTLayerRect>& pin_shape_list);
  std::vector<AccessPoint> getAccessPointListByPrefTrackGrid(std::vector<LayerRect>& legal_shape_list);
  std::vector<AccessPoint> getAccessPointListByCurrTrackGrid(std::vector<LayerRect>& legal_shape_list);
  std::vector<AccessPoint> getAccessPointListByTrackCenter(std::vector<LayerRect>& legal_shape_list);
  std::vector<AccessPoint> getAccessPointListByShapeCenter(std::vector<LayerRect>& legal_shape_list);
  void buildAccessPointList(PAModel& pa_model);
  void updatePAModel(PAModel& pa_model);

#if 1  // exhibit
  void plotPAModel(PAModel& pa_model);
  void reportPAModel(PAModel& pa_model);
  void reportSummary(PAModel& pa_model);
  void writePinCSV(PAModel& pa_model);
#endif
};

}  // namespace irt
