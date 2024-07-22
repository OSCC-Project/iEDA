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
#include "define.h"

namespace ito {
using ista::LibCell;
using ista::Pin;
using ista::StaSeqPathData;

class BufferedOption;

using BufferedOptionSeq = vector<BufferedOption *>;

enum class BufferedOptionType { kBuffer, kBranch, kSink, kWire };

class BufferedOption {
 public:
  BufferedOption(BufferedOptionType option_type) : _option_type(option_type) {}
  ~BufferedOption() = default;

  BufferedOptionType get_type() const { return _option_type; }

  float get_cap() const { return _cap; }
  void set_cap(float cap) { _cap = cap; }

  double get_req() { return _req; }
  void set_req(double req) { _req = req; }

  TORequired get_required_arrival_time();

  TODelay get_delay_required() const { return _delay_required; }
  void set_delay_required(TODelay required_delay) { _delay_required = required_delay; }

  Point get_location() const { return _location; }
  void set_location(Point loc) { _location = loc; }

  LibCell* get_lib_cell_size() const { return _lib_cell_size; }
  void set_lib_cell_size(LibCell* cell) { _lib_cell_size = cell; }

  Pin* get_pin_loaded() const { return _pin_loaded; }
  void set_pin_loaded(Pin* pin) { _pin_loaded = pin; }
  BufferedOption* get_left() const { return _left; }
  void set_left(BufferedOption* left) { _left = left; }

  BufferedOption* get_right() const { return _left; }
  void set_right(BufferedOption* right) { _right = right; }

  void printBuffered(int level);
  void printTree(int level);

 private:
  BufferedOptionType _option_type;
  Point              _location;

  float _cap = 0.0;

  Pin *_pin_loaded = nullptr;

  TODelay _delay_required = 0.0;

  LibCell    *_lib_cell_size = nullptr;
  BufferedOption *_left = nullptr;
  BufferedOption *_right = nullptr;

  double _req = 0.0;
};

} // namespace ito
