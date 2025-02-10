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

#include <set>
#include <vector>

#include "DRCViolationType.h"

namespace idb {
class IdbLayer;
}

namespace idrc {

enum class Type
{
  kNone,
  kRect,
  kPolygon,
  kMax
};

class DrcViolation
{
 public:
  DrcViolation(idb::IdbLayer* layer, ViolationEnumType violation_type, Type type)
      : _layer(layer), _violation_type(violation_type), _type(type)
  {
  }
  ~DrcViolation()
  {
    _layer = nullptr;
    _net_ids.clear();
    _inst_ids.clear();
  }
  void set_net_ids(std::set<int> net_ids) { _net_ids = net_ids; }
  void set_inst_ids(std::set<int> inst_ids) { _inst_ids = inst_ids; }
  idb::IdbLayer* get_layer() { return _layer; }
  std::set<int>& get_net_ids() { return _net_ids; }
  std::set<int>& get_inst_ids() { return _inst_ids; }
  Type get_type() { return _type; }
  bool is_rect() { return _type == Type::kRect; }
  bool is_polygon() { return _type == Type::kPolygon; }
  bool ignored();
  ViolationEnumType get_violation_type() { return _violation_type; }

 private:
  idb::IdbLayer* _layer;
  std::set<int> _net_ids;
  std::set<int> _inst_ids;
  ViolationEnumType _violation_type;
  Type _type;
};

class DrcViolationRect : public DrcViolation
{
 public:
  DrcViolationRect(idb::IdbLayer* layer, ViolationEnumType vio_type, int llx, int lly, int urx, int ury)
      : DrcViolation(layer, vio_type, Type::kRect), _llx(llx), _lly(lly), _urx(urx), _ury(ury)
  {
  }
  ~DrcViolationRect() {}

  int get_llx() { return _llx; }
  int get_lly() { return _lly; }
  int get_urx() { return _urx; }
  int get_ury() { return _ury; }

  /// check is line
  bool isLine()
  {
    if ((_llx == _urx && _lly != _ury) || (_llx != _urx && _lly == _ury)) {
      return true;
    } else {
      return false;
    }
  }

 private:
  int _llx;
  int _lly;
  int _urx;
  int _ury;
};

class DrcViolationPolygon : public DrcViolation
{
 public:
  DrcViolationPolygon(idb::IdbLayer* layer, ViolationEnumType vio_type) : DrcViolation(layer, vio_type, Type::kPolygon) {}
  ~DrcViolationPolygon() { _points.clear(); }

  std::vector<std::pair<int, int>>& get_points() { return _points; }
  void addPoints(int x, int y) { _points.emplace_back(std::make_pair(x, y)); }

 private:
  std::vector<std::pair<int, int>> _points;
};

}  // namespace idrc