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
