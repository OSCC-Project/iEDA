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
Required BufferedOption::get_required_arrival_time() { return _req - _required_delay; }

void BufferedOption::printBuffered(int level) {
  printTree(level);
  switch (_type) {
  case BufferedOptionType::kLoad: {
    break;
  }
  case BufferedOptionType::kBuffer:
  case BufferedOptionType::kWire: {
    _left->printBuffered(level + 1);
    break;
  }
  case BufferedOptionType::kJunction: {
    if (_left) {
      _left->printBuffered(level + 1);
    }
    if (_right) {
      _right->printBuffered(level + 1);
    }
    break;
  }
  }
}

void BufferedOption::printTree(int level) {
  switch (_type) {
  case BufferedOptionType::kLoad: {
    printf("%*s load %s (%d, %d) cap %f req %lf\n", level, "", _load_pin->get_name(),
           _location.get_x(), _location.get_y(), _cap, _req);
    break;
  }
  case BufferedOptionType::kBuffer: {
    printf("%*s buffer (%d, %d) %s cap %f req %lf\n", level, "", _location.get_x(),
           _location.get_y(), _buffer_cell->get_cell_name(), _cap,
           get_required_arrival_time());
    break;
  }
  case BufferedOptionType::kWire: {
    printf("%*s wire (%d, %d) cap %f req %lf\n", level, "", _location.get_x(),
           _location.get_y(), _cap, get_required_arrival_time());
    break;
  }
  case BufferedOptionType::kJunction: {
    printf("%*s junction (%d, %d) cap %f req %lf\n", level, "", _location.get_x(),
           _location.get_y(), _cap, get_required_arrival_time());
    break;
  }
  }
}

} // namespace ito