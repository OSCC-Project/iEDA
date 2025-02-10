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

#include "BufferedOption.h"

#include "api/TimingEngine.hh"

namespace ito {
TORequired BufferedOption::get_required_arrival_time()
{
  return _req - _delay_required;
}

void BufferedOption::printBuffered(int level)
{
  printTree(level);
  if (_left) {
    switch (_option_type) {
      case BufferedOptionType::kBuffer:
      case BufferedOptionType::kWire:
      case BufferedOptionType::kBranch:
        _left->printBuffered(level + 1);
        break;
      default:
        break;
    }
  }
  if (_right && _option_type == BufferedOptionType::kBranch) {
    _right->printBuffered(level + 1);
  }
}

void BufferedOption::printTree(int level)
{
  const char* typeStr = nullptr;
  string specificStr = "---";

  switch (_option_type) {
    case BufferedOptionType::kSink: {
      typeStr = "load";
      specificStr = _pin_loaded->getFullName();
      break;
    }
    case BufferedOptionType::kBuffer: {
      typeStr = "buffer";
      specificStr = _lib_cell_size->get_cell_name();
      break;
    }
    case BufferedOptionType::kWire: {
      typeStr = "wire";
      break;
    }
    case BufferedOptionType::kBranch: {
      typeStr = "junction";
      break;
    }
  }

  if (typeStr) {
    printf("%*s %s loc (%d, %d), %s, cap_load %f, required_arrival_time %lf\n", level, "", typeStr, _location.get_x(), _location.get_y(),
           specificStr.c_str(), _cap, get_required_arrival_time());
  }
}

}  // namespace ito