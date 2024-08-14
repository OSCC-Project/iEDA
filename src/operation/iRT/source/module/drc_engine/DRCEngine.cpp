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
#include "DRCEngine.hpp"

#include "GDSPlotter.hpp"
#include "RTInterface.hpp"
#include "Utility.hpp"

namespace irt {

// public

void DRCEngine::initInst()
{
  if (_de_instance == nullptr) {
    _de_instance = new DRCEngine();
  }
}

DRCEngine& DRCEngine::getInst()
{
  if (_de_instance == nullptr) {
    RTLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_de_instance;
}

void DRCEngine::destroyInst()
{
  if (_de_instance != nullptr) {
    delete _de_instance;
    _de_instance = nullptr;
  }
}

// function

std::vector<Violation> DRCEngine::getViolationList(std::string top_name, std::vector<std::pair<EXTLayerRect*, bool>>& env_shape_list,
                                                   std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>>& net_pin_shape_map,
                                                   std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_result_map, std::string stage)
{
  return RTI.getViolationList(env_shape_list, net_pin_shape_map, net_result_map, stage);
}

// private

DRCEngine* DRCEngine::_de_instance = nullptr;

}  // namespace irt
