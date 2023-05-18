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

#include "FPPin.hh"

namespace ipl::imp {

FPPin::FPPin()
{
  _name = "";
  _is_io_pin = false;
  _coordinate = new Coordinate();
  _instance = nullptr;
  _net = nullptr;
  _weight = 1;
}

FPPin::~FPPin()
{
  if (_coordinate != nullptr) {
    delete _coordinate;
    _coordinate = nullptr;
  }
}

int32_t FPPin::get_x()
{
  if (!_is_io_pin) {
    return _instance->get_center_x() + get_offset_x();
  } else {
    return _coordinate->_x;
  }
}

int32_t FPPin::get_y()
{
  if (!_is_io_pin) {
    return _instance->get_center_y() + get_offset_y();
  } else {
    return _coordinate->_y;
  }
}

int32_t FPPin::get_offset_x()
{
  Orient orient = _instance->get_orient();
  if (orient == Orient::N || orient == Orient::FN) {
    return _coordinate->_x;
  } else if (orient == Orient::E || orient == Orient::FE) {
    return _coordinate->_y;
  } else if (orient == Orient::S || orient == Orient::FS) {
    return -(_coordinate->_x);
  } else if (orient == Orient::W || orient == Orient::FW) {
    return -(_coordinate->_y);
  }
  return 0;
}

int32_t FPPin::get_offset_y()
{
  Orient orient = _instance->get_orient();
  if (orient == Orient::N || orient == Orient::FN) {
    return _coordinate->_y;
  } else if (orient == Orient::E || orient == Orient::FW) {
    return -(_coordinate->_x);
  } else if (orient == Orient::S || orient == Orient::FS) {
    return -(_coordinate->_y);
  } else if (orient == Orient::W || orient == Orient::FE) {
    return _coordinate->_x;
  }
  return 0;
}

}  // namespace ipl::imp