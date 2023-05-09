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

#include <assert.h>

namespace idb {

// All operation is under database unit(dbu).
//
// Default:
//  1 user unit = 1 micron
//  1 user unit = 1000 dbu
//  Hence GdsUnit.in_user = 0.001, GdsUnit.in_meter = 1e-9;
//
// Typically:
//  GdsUnit._in_user <= 1,
//  since you use more than 1 database unit per user unit.
//
// Error when setting GdsUnit._in_user = 0 or GdsUnit._in_meter = 0
class GdsUnit
{
 public:
  // constructor
  GdsUnit() : _in_user(0.001f), _in_meter(1e-9) {}
  explicit GdsUnit(float in_user, float in_meter);

  // getter
  float dbu_in_user() const;
  float dbu_in_meter() const;
  float user_unit_in_meter() const;
  float meter_in_user_unit() const;

  // setter
  void set_in_user(float v);
  void set_in_meter(float v);

  // operator
  GdsUnit& operator=(const GdsUnit&);

 private:
  float _in_user;   // the first number is the size of a database unit in user units
  float _in_meter;  // the second number is the size of a database unit in meters
};

/////////// inline ///////////

// @param in_user   the size of 1 dbu in user units
// @param in_meter  the size of 1 dbu in meters
inline GdsUnit::GdsUnit(float in_user, float in_meter)
{
  assert(in_user);
  assert(in_meter);

  _in_user = in_user;
  in_meter = in_meter;
}

inline void GdsUnit::set_in_user(float v)
{
  assert(v);
  _in_user = v;
}

inline void GdsUnit::set_in_meter(float v)
{
  assert(v);
  _in_meter = v;
}

inline float GdsUnit::dbu_in_user() const
{
  return _in_user;
}

inline float GdsUnit::dbu_in_meter() const
{
  return _in_meter;
}

// the size of a user unit in meter
inline float GdsUnit::user_unit_in_meter() const
{
  return _in_meter / _in_user;
}

// the size of a meter in user unit
inline float GdsUnit::meter_in_user_unit() const
{
  return _in_user / _in_meter;
}

inline GdsUnit& GdsUnit::operator=(const GdsUnit& rhs)
{
  _in_user = rhs._in_user;
  _in_meter = rhs._in_meter;

  return *this;
}

}  // namespace idb
