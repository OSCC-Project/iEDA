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

#pragma once

#include "DbInterface.h"
#include "Point.h"
#include "ids.hpp"

namespace ito {
using ista::LibCell;
using ista::Pin;
using ista::StaSeqPathData;

class BufferedOption;

using BufferedOptionSeq = vector<BufferedOption *>;

enum class BufferedOptionType { kBuffer, kJunction, kLoad, kWire };

class BufferedOption {
 public:
  BufferedOption(BufferedOptionType type, Point location, float cap, ista::Pin *load_pin,
                 TODelay required_delay, LibCell *buffer, BufferedOption *left,
                 BufferedOption *right, double req)
      : _type(type), _location(location), _cap(cap), _load_pin(load_pin),
        _required_delay(required_delay), _buffer_cell(buffer), _left(left), _right(right),
        _req(req) {}
  ~BufferedOption() = default;

  BufferedOptionType get_type() const { return _type; }

  float get_cap() const { return _cap; }

  double get_req() { return _req; }

  TORequired get_required_arrival_time();

  TODelay get_required_delay() const { return _required_delay; }

  Point get_location() const { return _location; }

  LibCell *get_buffer_cell() const { return _buffer_cell; }

  Pin *get_load_pin() const { return _load_pin; }

  BufferedOption *get_left() const { return _left; }

  BufferedOption *get_right() const { return _right; }

  void printBuffered(int level);
  void printTree(int level);

 private:
  BufferedOptionType _type;
  Point              _location;

  float _cap = 0.0;

  Pin *_load_pin = nullptr;

  TODelay _required_delay = 0.0;

  LibCell *       _buffer_cell = nullptr;
  BufferedOption *_left = nullptr;
  BufferedOption *_right = nullptr;

  double _req = 0.0;
};

} // namespace ito
