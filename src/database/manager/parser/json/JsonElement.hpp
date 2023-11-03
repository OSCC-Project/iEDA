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

#include <iostream>
#include <map>
#include <string>

#include "JsonELFlags.hpp"
#include "JsonTypedef.h"
#include "JsonXY.hpp"

namespace idb {

enum class JsonElemType
{
  kElement,
  kBoundary,
  kPath,
  kSref,
  kAref,
  kText,
  kNode,
  kBox
};

class JsonElemBase
{
 public:
  // constructor
  explicit JsonElemBase(JsonElemType t);
  virtual ~JsonElemBase() = default;

  // getter
  const char* get_property(int16_t) const;
  size_t get_property_num() const;
  JsonElemType get_elem_type() const;
  const JsonXY& get_xy() const { return _xy; }
  const std::map<int16_t, std::string>& get_property_map() const;
  JsonELFlags get_flags() const;
  bool is_external() const;
  bool is_template_data() const;
  JsonPlexType get_plex() const;

  // setter
  void set_property(int16_t, const std::string&);
  void add_coord(int32_t x, int32_t y);
  void add_coord(const JXYCoordinate& c);
  void set_flags(const JsonELFlags&);
  void set_external(bool);
  void set_template_data(bool);
  void set_plex(JsonPlexType);

  // operator
  JsonElemBase& operator=(const JsonElemBase&);

  // function
  void remove_property(int16_t);
  void reset_base();
  virtual void reset() = 0;

 private:
  // members
  JsonELFlags _flags;
  JsonPlexType _plex;  //
  std::map<int16_t, std::string> _property_map;
  JsonElemType _type;
  JsonXY _xy;
};

class JsonElement : public JsonElemBase
{
 public:
  JsonElement() : JsonElemBase(JsonElemType::kElement) {}
  void reset() override;
};

///////////// inline ////////////

inline JsonElemBase::JsonElemBase(JsonElemType t) : _flags(), _plex(), _property_map(), _type(t), _xy()
{
}

inline const char* JsonElemBase::get_property(int16_t attr) const
{
  auto it = _property_map.find(attr);
  return it != _property_map.end() ? it->second.c_str() : "";
}

inline void JsonElemBase::set_property(int16_t attr, const std::string& v)
{
  _property_map[attr] = v;
}

inline size_t JsonElemBase::get_property_num() const
{
  return _property_map.size();
}

inline JsonElemType JsonElemBase::get_elem_type() const
{
  return _type;
}

inline void JsonElemBase::remove_property(int16_t attr)
{
  _property_map.erase(attr);
}

inline void JsonElemBase::add_coord(int32_t x, int32_t y)
{
  _xy.add_coord(x, y);
}

inline void JsonElemBase::add_coord(const JXYCoordinate& c)
{
  _xy.add_coord(c);
}

inline JsonElemBase& JsonElemBase::operator=(const JsonElemBase& rhs)
{
  _property_map = rhs._property_map;
  _type = rhs._type;
  _xy = rhs._xy;

  return *this;
}

inline const std::map<int16_t, std::string>& JsonElemBase::get_property_map() const
{
  return _property_map;
}

inline JsonELFlags JsonElemBase::get_flags() const
{
  return _flags;
}

inline void JsonElemBase::set_flags(const JsonELFlags& f)
{
  _flags = f;
}

inline void JsonElemBase::set_external(bool b)
{
  _flags.set_is_external(b);
}

inline void JsonElemBase::set_template_data(bool b)
{
  _flags.set_is_template(b);
}

inline bool JsonElemBase::is_external() const
{
  return _flags.is_external();
}

inline bool JsonElemBase::is_template_data() const
{
  return _flags.is_template();
}

inline void JsonElemBase::reset_base()
{
  _flags.reset();
  _property_map.clear();
  _xy.clear();
  _flags.reset();
}

// A unique positive number which is common to all elementsof the plex to which this element belongs.
inline JsonPlexType JsonElemBase::get_plex() const
{
  return _plex;
}

inline void JsonElemBase::set_plex(JsonPlexType plex)
{
  _plex = plex;
}

inline void JsonElement::reset()
{
  reset_base();
}

}  // namespace idb