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

namespace icts {

struct ZstTag
{
};
struct BstTag
{
};
struct UstTag
{
};

enum class DelayModel
{
  kLinear = 0,
  kElmore = 1
};
class Params
{
 public:
  Params() : _model(DelayModel::kLinear), _unit_res(0), _unit_cap(0) {}
  Params(DelayModel model, int db_unit, double unit_res = 1, double unit_cap = 1)
      : _model(model), _db_unit(db_unit), _unit_res(unit_res), _unit_cap(unit_cap)
  {
  }

  DelayModel get_delay_model() const { return _model; }
  double get_unit_res() const { return _unit_res; }
  double get_unit_cap() const { return _unit_cap; }
  int get_db_unit() const { return _db_unit; }

  void set_delay_model(DelayModel model) { _model = model; }
  void set_unit_res(double unit_res) { _unit_res = unit_res; }
  void set_unit_cap(double unit_cap) { _unit_cap = unit_cap; }
  void set_db_unit(int db_unit) { _db_unit = db_unit; }

 private:
  DelayModel _model;
  int _db_unit;
  double _unit_res;
  double _unit_cap;
};

class ZstParams : public Params
{
 public:
  typedef ZstTag DmeTag;

  ZstParams() : Params() {}
  ZstParams(DelayModel model, int db_unit, double unit_res = 0, double unit_cap = 0) : Params(model, db_unit, unit_res, unit_cap) {}
};

enum class BstType
{
  kBME = 0,
  kIME = 1
};
class BstParams : public Params
{
 public:
  typedef BstTag DmeTag;

  BstParams() : Params(), _type(BstType::kBME), _skew_bound(0) {}
  BstParams(BstType type, int db_unit, double skew_bound, DelayModel model, double unit_res = 0, double unit_cap = 0)
      : Params(model, db_unit, unit_res, unit_cap), _type(type), _skew_bound(skew_bound)
  {
  }

  BstType get_bst_type() const { return _type; }
  void set_bst_type(BstType type) { _type = type; }

  double get_skew_bound() const { return _skew_bound; }
  void set_skew_bound(double skew_bound) { _skew_bound = skew_bound; }

 private:
  BstType _type;
  double _skew_bound;
};

class UstParams : public Params
{
 public:
  typedef UstTag DmeTag;
  UstParams() : Params() {}
  UstParams(DelayModel model, int db_unit, double skew_bound, double unit_res = 0, double unit_cap = 0)
      : Params(model, db_unit, unit_res, unit_cap), _skew_bound(skew_bound)
  {
  }
  double get_skew_bound() const { return _skew_bound; }
  void set_skew_bound(double skew_bound) { _skew_bound = skew_bound; }

 private:
  double _skew_bound;
};
}  // namespace icts