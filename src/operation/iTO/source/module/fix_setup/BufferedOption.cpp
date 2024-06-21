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

/////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2019, The Regents of the University of California
// All rights reserved.
//
// BSD 3-Clause License
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////////

#include "BufferedOption.h"
#include "api/TimingEngine.hh"

namespace ito {
TORequired BufferedOption::get_required_arrival_time() { return _req - _required_delay; }

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
    printf("%*s load %s (%d, %d), load_cap %f, required_arrival_time %lf\n", level, "", _load_pin->getFullName().c_str(),
           _location.get_x(), _location.get_y(), _cap, _req);
    break;
  }
  case BufferedOptionType::kBuffer: {
    printf("%*s buffer (%d, %d), %s load_cap %f, required_arrival_time %lf\n", level, "", _location.get_x(),
           _location.get_y(), _buffer_cell->get_cell_name(), _cap,
           get_required_arrival_time());
    break;
  }
  case BufferedOptionType::kWire: {
    printf("%*s wire (%d, %d), load_cap %f, required_arrival_time %lf\n", level, "", _location.get_x(),
           _location.get_y(), _cap, get_required_arrival_time());
    break;
  }
  case BufferedOptionType::kJunction: {
    printf("%*s junction (%d, %d), load_cap %f, required_arrival_time %lf\n", level, "", _location.get_x(),
           _location.get_y(), _cap, get_required_arrival_time());
    break;
  }
  }
}

} // namespace ito