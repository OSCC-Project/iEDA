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

#include "DRCInterface.hpp"

#include "DataManager.hpp"

namespace idrc {

// public

DRCInterface& DRCInterface::getInst()
{
  if (_drc_interface_instance == nullptr) {
    _drc_interface_instance = new DRCInterface();
  }
  return *_drc_interface_instance;
}

void DRCInterface::destroyInst()
{
  if (_drc_interface_instance != nullptr) {
    delete _drc_interface_instance;
    _drc_interface_instance = nullptr;
  }
}

#if 1  // 外部调用DRC的API

#if 1  // iDRC

void DRCInterface::initDRC()
{
  // 初始化规则
}

void DRCInterface::checkDef()
{
  std::vector<idb::IdbLayerShape*> idb_env_shape_list;
  std::map<int32_t, std::vector<idb::IdbLayerShape*>> idb_net_pin_shape_map;
  std::map<int32_t, std::vector<idb::IdbRegularWireSegment*>> idb_net_result_map;
  // create data
  {
  }
  // check
  getViolationList(idb_env_shape_list, idb_net_pin_shape_map, idb_net_result_map);
  // free memory
  {
    for (idb::IdbLayerShape* idb_env_shape : idb_env_shape_list) {
      delete idb_env_shape;
      idb_env_shape = nullptr;
    }
    for (auto& [net_idx, pin_shape_list] : idb_net_pin_shape_map) {
      for (idb::IdbLayerShape* pin_shape : pin_shape_list) {
        delete pin_shape;
        pin_shape = nullptr;
      }
    }
    for (auto& [net_idx, segment_list] : idb_net_result_map) {
      for (idb::IdbRegularWireSegment* segment : segment_list) {
        delete segment;
        segment = nullptr;
      }
    }
  }
}

std::vector<ids::Violation> DRCInterface::getViolationList(std::vector<idb::IdbLayerShape*>& idb_env_shape_list,
                                                           std::map<int32_t, std::vector<idb::IdbLayerShape*>>& idb_net_pin_shape_map,
                                                           std::map<int32_t, std::vector<idb::IdbRegularWireSegment*>>& idb_net_result_map)
{
  return {};
}

void DRCInterface::destroyDRC()
{
  // 销毁规则
}

#endif

#endif

// private

DRCInterface* DRCInterface::_drc_interface_instance = nullptr;

}  // namespace idrc
