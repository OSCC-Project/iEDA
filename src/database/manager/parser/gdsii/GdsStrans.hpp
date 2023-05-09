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

#include <iostream>

namespace idb {

class GdsStrans
{
 public:
  GdsStrans();

  // getter
  bool reflection() const;
  bool abs_mag() const;
  bool abs_angle() const;

  // function
  void reset();

  // members
  uint16_t bit_flag;  // bit0: the leftmost -> bit15: the rightmost
  double mag;         // magnification factor.
  double angle;       // The angle of rotation is measured in degrees and in the counterclockwise direction
};

////////// inline /////////
inline GdsStrans::GdsStrans() : bit_flag(0), mag(1), angle(0)
{
}

// If bit 0 is set, the element is reflected about the X-axis before angular rotation.
inline bool GdsStrans::reflection() const
{
  return bit_flag & 0x80;
}

// Bit 13 flags absolute magnification.
inline bool GdsStrans::abs_mag() const
{
  return bit_flag & 0x04;
}

// Bit 14 flags absolute angle.
inline bool GdsStrans::abs_angle() const
{
  return bit_flag & 0x02;
}

inline void GdsStrans::reset()
{
  bit_flag = 0;
  mag = 1;
  angle = 0;
}

}  // namespace idb
