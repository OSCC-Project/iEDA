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

#include <stdint.h>

#include "drc_basic_point.h"
#include "idrc_data.h"

namespace idrc {

class DrcSegmentType
{
 public:
  enum Type : uint64_t
  {
    kNone = 0,
    kOverlap = 1,
    kConvexStartEdge = 2,
    kConcaveStartEdge = 4,
    kTurnOutEdge = 8,
    kTurnInEdge = 16,
    kConvexEndEdge = 32,
    kConcaveEndEdge = 64,
    kSlefSpacing = 128,
    kMutualSpacing = 256,
    kWidth = 512
  };

  DrcSegmentType(Type type = kNone) : _type(type) {}
  ~DrcSegmentType() {}

  Type get_type() { return _type; }

  bool isEdge() { return _type & (kConvexStartEdge | kConcaveStartEdge | kTurnOutEdge | kTurnInEdge | kConvexEndEdge | kConcaveEndEdge); }

  bool operator==(const DrcSegmentType& type) const { return _type == type._type; }

 private:
  Type _type;
};

class DrcBasicSegment
{
 public:
  DrcBasicSegment(DrcCoordinate begin, DrcCoordinate end) : _begin(begin), _end(end) {}
  DrcBasicSegment(int lb_x, int lb_y, int ur_x, int ur_y) : _begin(lb_x, lb_y), _end(ur_x, ur_y) {}
  ~DrcBasicSegment() {}

  DrcCoordinate get_begin() { return _begin; }
  DrcCoordinate get_end() { return _end; }

 private:
  DrcCoordinate _begin;
  DrcCoordinate _end;
};

class DrcSegmentWithType : public DrcBasicSegment
{
 public:
  DrcSegmentWithType(DrcCoordinate start, DrcCoordinate end, DrcSegmentType type = DrcSegmentType::kNone)
      : DrcBasicSegment(start, end), _type(type)
  {
  }
  ~DrcSegmentWithType() {}

 private:
  DrcSegmentType _type;
};

}  // namespace idrc
