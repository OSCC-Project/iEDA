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
#include "DPInstance.hh"

namespace ipl {

DPInstance::DPInstance(std::string name)
    : _dp_inst_id(-1),
      _name(name),
      _master(nullptr),
      _belong_region(nullptr),
      _cluster_internal_id(INT32_MIN),
      _belong_cluster(nullptr),
      _weight(1.0)
{
}

DPInstance::~DPInstance()
{
}

void DPInstance::updateCoordi(int32_t llx, int32_t lly)
{
  _shape.set_rectangle(llx, lly, llx + _shape.get_width(), lly + _shape.get_height());
  updatePinsCoordi();
}

std::pair<int32_t, int32_t> DPInstance::calInstPinModifyOffest(DPPin* pin)
{
  auto* pin_inst = pin->get_instance();
  if (!pin_inst || !(pin_inst == this)) {
    LOG_WARNING << " Request pin is not of Instance";
    return std::make_pair(pin->get_offset_x(), pin->get_offset_y());
  }

  int32_t origin_offset_x = pin->get_offset_x();
  int32_t origin_offset_y = pin->get_offset_y();

  int32_t modify_offset_x = 0;
  int32_t modify_offset_y = 0;
  if (_orient == Orient::kN_R0) {
    modify_offset_x = origin_offset_x;
    modify_offset_y = origin_offset_y;
  } else if (_orient == Orient::kW_R90) {
    modify_offset_x = (-1) * origin_offset_y;
    modify_offset_y = origin_offset_x;
  } else if (_orient == Orient::kS_R180) {
    modify_offset_x = (-1) * origin_offset_x;
    modify_offset_y = (-1) * origin_offset_y;
  } else if (_orient == Orient::kFW_MX90) {
    modify_offset_x = origin_offset_y;
    modify_offset_y = origin_offset_x;
  } else if (_orient == Orient::kFN_MY) {
    modify_offset_x = (-1) * origin_offset_x;
    modify_offset_y = origin_offset_y;
  } else if (_orient == Orient::kFE_MY90) {
    modify_offset_x = (-1) * origin_offset_y;
    modify_offset_y = (-1) * origin_offset_x;
  } else if (_orient == Orient::kFS_MX) {
    modify_offset_x = origin_offset_x;
    modify_offset_y = (-1) * origin_offset_y;
  } else if (_orient == Orient::kE_R270) {
    modify_offset_x = origin_offset_y;
    modify_offset_y = (-1) * origin_offset_x;
  } else {
    LOG_WARNING << pin_inst->get_name() + " has not set the orient!";
  }

  return std::make_pair(modify_offset_x, modify_offset_y);
}

void DPInstance::updatePinsCoordi()
{
  int32_t center_x = _shape.get_ll_x() + _master->get_width() / 2;
  int32_t center_y = _shape.get_ll_y() + _master->get_height() / 2;

  for (auto* pin : _pin_list) {
    std::pair<int32_t, int32_t> modify_offset = calInstPinModifyOffest(pin);

    pin->set_x_coordi(center_x + modify_offset.first);
    pin->set_y_coordi(center_y + modify_offset.second);
  }
}

int64_t DPInstance::computeDisplacement() const
{
  const int64_t pos_x = get_shape().get_ll_x();
  const int64_t pos_y = get_shape().get_ll_y();
  const int64_t init_x = get_origin_shape().get_ll_x();
  const int64_t init_y = get_origin_shape().get_ll_y();
  return std::abs(pos_x - init_x) + std::abs(pos_y - init_y);
}


}  // namespace ipl