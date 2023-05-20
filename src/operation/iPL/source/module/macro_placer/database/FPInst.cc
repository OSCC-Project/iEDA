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

#include "FPInst.hh"

namespace ipl::imp {

FPInst::FPInst()
{
  _name = "";
  _index = -1;
  _type = InstType::kStd_cell;
  _fixed = false;
  _halo_x = 0;
  _halo_y = 0;
  _has_halo = false;
  _orient = Orient::kN;
  _main_orient = true;
  _align_flag = false;
}

FPInst::~FPInst()
{
  _pin_list.clear();
}

void FPInst::set_orient(Orient orient)
{
  _orient = orient;
  if (_orient == Orient::kN || _orient == Orient::kFN || _orient == Orient::kS || _orient == Orient::kFS) {
    _main_orient = true;
  } else {
    _main_orient = false;
  }
}

uint32_t FPInst::get_width() const
{
  {
    if (_main_orient) {
      return _width;
    } else {
      return _height;
    }
  }
}

uint32_t FPInst::get_height() const
{
  if (_main_orient) {
    return _height;
  } else {
    return _width;
  }
}

void FPInst::addHalo()
{
  if (!_has_halo) {
    _width += 2 * _halo_x;
    _height += 2 * _halo_y;
    _coordinate->set_x(_coordinate->get_x() - _halo_x);
    _coordinate->set_y(_coordinate->get_y() - _halo_y);
    _has_halo = true;
  }
}

void FPInst::deleteHalo()
{
  if (_has_halo) {
    _width -= 2 * _halo_x;
    _height -= 2 * _halo_y;
    _coordinate->set_x(_coordinate->get_x() + _halo_x);
    _coordinate->set_y(_coordinate->get_y() + _halo_y);
    _has_halo = false;
  }
}

std::string FPInst::get_orient_name()
{
  switch (_orient) {
    case Orient::kN:
      return "N,R0";
    case Orient::kS:
      return "S,R180";
    case Orient::kW:
      return "W,R90";
    case Orient::kE:
      return "E,R270";
    case Orient::kFN:
      return "FN,MY";
    case Orient::kFS:
      return "FS,MX";
    case Orient::kFW:
      return "FW,MX90";
    case Orient::kFE:
      return "FE,MY90";

    default:
      return "kNone,kNone";
      break;
  }
}

}  // namespace ipl::imp