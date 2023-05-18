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
  _type = InstType::STD;
  _fixed = false;
  _halo_x = 0;
  _halo_y = 0;
  _has_halo = false;
  _orient = Orient::N;
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
  if (_orient == Orient::N || _orient == Orient::FN || _orient == Orient::S || _orient == Orient::FS) {
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
    _coordinate->_x -= _halo_x;
    _coordinate->_y -= _halo_y;
    _has_halo = true;
  }
}

void FPInst::deleteHalo()
{
  if (_has_halo) {
    _width -= 2 * _halo_x;
    _height -= 2 * _halo_y;
    _coordinate->_x += _halo_x;
    _coordinate->_y += _halo_y;
    _has_halo = false;
  }
}

string FPInst::get_orient_name()
{
  switch (_orient) {
    case Orient::N:
      return "N,R0";
    case Orient::S:
      return "S,R180";
    case Orient::W:
      return "W,R90";
    case Orient::E:
      return "E,R270";
    case Orient::FN:
      return "FN,MY";
    case Orient::FS:
      return "FS,MX";
    case Orient::FW:
      return "FW,MX90";
    case Orient::FE:
      return "FE,MY90";

    default:
      return "kNone,kNone";
      break;
  }
}

}  // namespace ipl::imp