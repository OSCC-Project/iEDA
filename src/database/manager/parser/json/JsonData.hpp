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

#include "JsonFormat.hpp"
#include "JsonStruct.hpp"
#include "JsonTimestamp.hpp"
#include "JsonUnit.hpp"

namespace idb {

// This module is referred to
// https://www.boolean.klaasholwerda.nl/interface/bnf/jsonformat.html
class JsonData
{
 public:
  // constructor
  JsonData();
  ~JsonData();

  // getter
  JsonHeader get_header() const;
  time_t get_bgn_lib() const;
  time_t get_last_mod() const;
  std::string get_lib_name() const;
  JsonUnit& get_unit();
  const std::vector<JsonStruct*>& get_struct_list() const;
  std::vector<std::string> get_ref_libs() const;
  std::vector<std::string> get_fonts() const;
  std::string get_attrtable() const;
  JsonGenerations get_generations() const;
  JsonFormat get_format() const;
  JsonStruct* get_top_struct() { return _top_struct; }

  // setter
  void set_header(JsonHeader);
  void set_lib_timestamp(time_t b, time_t l);
  void set_lib_name(const std::string&);
  void set_unit(float in_user, float in_meter);
  void set_unit(const JsonUnit&);
  void add_struct(JsonStruct*);
  void add_struct(const JsonStruct&);
  void add_ref_lib(const std::string&);
  void add_font(const std::string&);
  void set_attrtable(const std::string&);
  void set_generations(JsonGenerations);
  void set_format(const JsonFormat&);
  void set_format(JsonFormatType, const std::string&);
  void set_top_struct(JsonStruct* top_struct) { _top_struct = top_struct; }

  // function
  void clear_struct_list();
  bool is_full() { 
    return static_cast<int>(_struct_list.size()) == max_num ? true : false;
  }

 private:
  const int max_num = 10000;
  // members
  JsonHeader _header;                   // version number
  JsonTimestamp _ts;                    // library timestamp
  std::string _name;                   // library name
  std::vector<std::string> _ref_libs;  // reference libraries
  std::vector<std::string> _fonts;
  std::string _attrtable;
  JsonGenerations _generations;
  JsonFormat _format;
  JsonUnit _unit;
  std::vector<JsonStruct*> _struct_list;
  JsonStruct* _top_struct = nullptr;
};

///////////// inline ////////

inline JsonHeader JsonData::get_header() const
{
  return _header;
}

inline time_t JsonData::get_bgn_lib() const
{
  return _ts.beg;
}

inline time_t JsonData::get_last_mod() const
{
  return _ts.last;
}

inline std::string JsonData::get_lib_name() const
{
  return _name;
}

inline JsonUnit& JsonData::get_unit()
{
  return _unit;
}

inline const std::vector<JsonStruct*>& JsonData::get_struct_list() const
{
  return _struct_list;
}

inline void JsonData::set_header(JsonHeader h)
{
  _header = h;
}

// @param b beginning of the library creation
// @param l last modification of the library
inline void JsonData::set_lib_timestamp(time_t b, time_t l)
{
  _ts.beg = b;
  _ts.last = l;
}

inline void JsonData::set_lib_name(const std::string& s)
{
  _name = s;
}

inline void JsonData::set_unit(float in_user, float in_meter)
{
  _unit.set_in_user(in_user);
  _unit.set_in_meter(in_meter);
}

inline void JsonData::set_unit(const JsonUnit& u)
{
  _unit = u;
}

inline void JsonData::add_struct(JsonStruct* s)
{
  if (s)
    _struct_list.emplace_back(s);
}

inline void JsonData::add_struct(const JsonStruct& s)
{
  auto cpy = new JsonStruct(s);
  add_struct(cpy);
}

inline std::vector<std::string> JsonData::get_ref_libs() const
{
  return _ref_libs;
}

inline void JsonData::add_ref_lib(const std::string& s)
{
  _ref_libs.emplace_back(s);
}

inline std::vector<std::string> JsonData::get_fonts() const
{
  return _fonts;
}

inline void JsonData::add_font(const std::string& s)
{
  _fonts.emplace_back(s);
}

inline std::string JsonData::get_attrtable() const
{
  return _attrtable;
}

inline void JsonData::set_attrtable(const std::string& s)
{
  _attrtable = s;
}

inline JsonGenerations JsonData::get_generations() const
{
  return _generations;
}

inline void JsonData::set_generations(JsonGenerations v)
{
  _generations = v;
}

inline JsonFormat JsonData::get_format() const
{
  return _format;
}

inline void JsonData::set_format(const JsonFormat& v)
{
  _format = v;
}

inline void JsonData::set_format(JsonFormatType t, const std::string& mask)
{
  _format.type = t;
  _format.mask = mask;
}

}  // namespace idb