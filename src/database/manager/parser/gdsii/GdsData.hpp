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
#include <time.h>

#include <string>
#include <vector>

#include "GdsFormat.hpp"
#include "GdsStruct.hpp"
#include "GdsTimestamp.hpp"
#include "GdsUnit.hpp"

namespace idb {

// This module is referred to
// https://www.boolean.klaasholwerda.nl/interface/bnf/gdsformat.html
class GdsData
{
 public:
  // constructor
  GdsData();
  ~GdsData();

  // getter
  GdsHeader get_header() const;
  time_t get_bgn_lib() const;
  time_t get_last_mod() const;
  std::string get_lib_name() const;
  GdsUnit& get_unit();
  const std::vector<GdsStruct*>& get_struct_list() const;
  std::vector<std::string> get_ref_libs() const;
  std::vector<std::string> get_fonts() const;
  std::string get_attrtable() const;
  GdsGenerations get_generations() const;
  GdsFormat get_format() const;
  GdsStruct* get_top_struct() { return _top_struct; }

  // setter
  void set_header(GdsHeader);
  void set_lib_timestamp(time_t b, time_t l);
  void set_lib_name(const std::string&);
  void set_unit(float in_user, float in_meter);
  void set_unit(const GdsUnit&);
  void add_struct(GdsStruct*);
  void add_struct(const GdsStruct&);
  void add_ref_lib(const std::string&);
  void add_font(const std::string&);
  void set_attrtable(const std::string&);
  void set_generations(GdsGenerations);
  void set_format(const GdsFormat&);
  void set_format(GdsFormatType, const std::string&);
  void set_top_struct(GdsStruct* top_struct) { _top_struct = top_struct; }

  // function
  void clear_struct_list();
  bool is_full() { 
    return static_cast<int>(_struct_list.size()) == max_num ? true : false;
  }

 private:
  const int max_num = 10000;
  // members
  GdsHeader _header;                   // version number
  GdsTimestamp _ts;                    // library timestamp
  std::string _name;                   // library name
  std::vector<std::string> _ref_libs;  // reference libraries
  std::vector<std::string> _fonts;
  std::string _attrtable;
  GdsGenerations _generations;
  GdsFormat _format;
  GdsUnit _unit;
  std::vector<GdsStruct*> _struct_list;
  GdsStruct* _top_struct = nullptr;
};

///////////// inline ////////

inline GdsHeader GdsData::get_header() const
{
  return _header;
}

inline time_t GdsData::get_bgn_lib() const
{
  return _ts.beg;
}

inline time_t GdsData::get_last_mod() const
{
  return _ts.last;
}

inline std::string GdsData::get_lib_name() const
{
  return _name;
}

inline GdsUnit& GdsData::get_unit()
{
  return _unit;
}

inline const std::vector<GdsStruct*>& GdsData::get_struct_list() const
{
  return _struct_list;
}

inline void GdsData::set_header(GdsHeader h)
{
  _header = h;
}

// @param b beginning of the library creation
// @param l last modification of the library
inline void GdsData::set_lib_timestamp(time_t b, time_t l)
{
  _ts.beg = b;
  _ts.last = l;
}

inline void GdsData::set_lib_name(const std::string& s)
{
  _name = s;
}

inline void GdsData::set_unit(float in_user, float in_meter)
{
  _unit.set_in_user(in_user);
  _unit.set_in_meter(in_meter);
}

inline void GdsData::set_unit(const GdsUnit& u)
{
  _unit = u;
}

inline void GdsData::add_struct(GdsStruct* s)
{
  if (s)
    _struct_list.emplace_back(s);
}

inline void GdsData::add_struct(const GdsStruct& s)
{
  auto cpy = new GdsStruct(s);
  add_struct(cpy);
}

inline std::vector<std::string> GdsData::get_ref_libs() const
{
  return _ref_libs;
}

inline void GdsData::add_ref_lib(const std::string& s)
{
  _ref_libs.emplace_back(s);
}

inline std::vector<std::string> GdsData::get_fonts() const
{
  return _fonts;
}

inline void GdsData::add_font(const std::string& s)
{
  _fonts.emplace_back(s);
}

inline std::string GdsData::get_attrtable() const
{
  return _attrtable;
}

inline void GdsData::set_attrtable(const std::string& s)
{
  _attrtable = s;
}

inline GdsGenerations GdsData::get_generations() const
{
  return _generations;
}

inline void GdsData::set_generations(GdsGenerations v)
{
  _generations = v;
}

inline GdsFormat GdsData::get_format() const
{
  return _format;
}

inline void GdsData::set_format(const GdsFormat& v)
{
  _format = v;
}

inline void GdsData::set_format(GdsFormatType t, const std::string& mask)
{
  _format.type = t;
  _format.mask = mask;
}

}  // namespace idb