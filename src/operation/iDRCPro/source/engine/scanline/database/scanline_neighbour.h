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

#include "drc_basic_point.h"

namespace idrc {

class DrcBasicPoint;

class ScanlineDataType
{
 public:
  enum DataType : uint64_t
  {
    kNone = 0,
    kSpacing = 1,
    kWidth = 2,
    kInterSpacing = 4,
    kEdge = 8,
    kOverlap = 16
  };

  ScanlineDataType(uint64_t type) : _type(type) {}

  uint64_t getType()
  {
    // based on priority: overlap > edge > width > spacing
    if (hasType(kOverlap)) {
      return kOverlap;
    } else if (hasType(kEdge)) {
      return kEdge;
    } else if (hasType(kInterSpacing)) {
      return kInterSpacing;
    } else if (hasType(kWidth)) {
      return kWidth;
    } else if (hasType(kSpacing)) {
      return kSpacing;
    }
    return kNone;
  }
  bool isType(uint64_t type) { return getType() == type; }
  bool hasType(uint64_t type) { return (_type & type) > 0; }

  ScanlineDataType& operator+=(ScanlineDataType type)
  {
    _type |= type._type;
    return *this;
  }

  ScanlineDataType& operator-=(ScanlineDataType type)
  {
    _type &= ~type._type;
    return *this;
  }

 private:
  uint64_t _type = kNone;
};

class ScanlineNeighbour
{
 public:
  ScanlineNeighbour(ScanlineDataType type, DrcBasicPoint* neighbour) : _type(type), _neighbour(neighbour) {}

  // getter
  ScanlineDataType get_type() { return _type; }
  DrcBasicPoint* get_point() { return _neighbour; }
  bool is_spacing() { return _type.isType(ScanlineDataType::kSpacing); }
  bool is_width() { return _type.isType(ScanlineDataType::kWidth); }
  bool is_overlap() { return _type.isType(ScanlineDataType::kOverlap); }
  bool is_edge() { return _type.isType(ScanlineDataType::kEdge); }

  // setter
  void set_type(ScanlineDataType type) { _type = type; }
  void add_type(ScanlineDataType type) { _type += type; }

 private:
  ScanlineDataType _type;

  DrcBasicPoint* _neighbour;
};

}  // namespace idrc
