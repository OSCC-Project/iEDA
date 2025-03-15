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

#include "DRCBox.hpp"
#include "DRCModel.hpp"
#include "DataManager.hpp"
#include "Module.hpp"

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
  // std::vector<idb::IdbLayerShape*> idb_env_shape_list;
  // std::map<int32_t, std::vector<idb::IdbLayerShape*>> idb_net_pin_shape_map;
  // std::map<int32_t, std::vector<idb::IdbRegularWireSegment*>> idb_net_result_map;
  // // create data
  // {
  // }
  // // check
  // getViolationList(idb_env_shape_list, idb_net_pin_shape_map, idb_net_result_map);
}

void DRCInterface::destroyDRC()
{
  // 销毁规则
}

std::vector<ids::Violation> DRCInterface::getViolationList(std::vector<ids::Shape> ids_shape_list)
{
  DRCModel drc_model = initDRCModel(ids_shape_list);
  DRCMOD.check(drc_model);
  return getViolationList(drc_model);
}

DRCModel DRCInterface::initDRCModel(std::vector<ids::Shape>& ids_shape_list)
{
  DRCModel drc_model;
  for (ids::Shape& ids_shape : ids_shape_list) {
    DRCShape drc_shape;
    drc_shape.set_net_idx(ids_shape.net_idx);
    drc_shape.set_ll(ids_shape.ll_x, ids_shape.ll_y);
    drc_shape.set_ur(ids_shape.ur_x, ids_shape.ur_y);
    drc_shape.set_layer_idx(ids_shape.layer_idx);
    drc_shape.set_is_routing(ids_shape.is_routing);
    drc_model.get_drc_shape_list().push_back(drc_shape);
  }
  return drc_model;
}

std::vector<ids::Violation> DRCInterface::getViolationList(DRCModel& drc_model)
{
  std::vector<ids::Violation> ids_violation_list;
  for (DRCBox& drc_box : drc_model.get_drc_box_list()) {
    for (Violation& violation : drc_box.get_violation_list()) {
      ids::Violation ids_violation;
      ids_violation.violation_type = GetViolationTypeName()(violation.get_violation_type());
      ids_violation.ll_x = violation.get_ll_x();
      ids_violation.ll_y = violation.get_ll_y();
      ids_violation.ur_x = violation.get_ur_x();
      ids_violation.ur_y = violation.get_ur_y();
      ids_violation.layer_idx = violation.get_layer_idx();
      ids_violation.is_routing = violation.get_is_routing();
      ids_violation.violation_net_set = violation.get_violation_net_set();
      ids_violation.required_size = violation.get_required_size();
      ids_violation_list.push_back(ids_violation);
    }
  }
  return ids_violation_list;
}

#endif

#endif

// private

DRCInterface* DRCInterface::_drc_interface_instance = nullptr;

}  // namespace idrc
