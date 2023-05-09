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

#include "GdsELFlags.hpp"
#include "GdsTypedef.h"
#include "GdsXY.hpp"

namespace idb {

enum class GdsElemType
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

class GdsElemBase
{
 public:
  // constructor
  explicit GdsElemBase(GdsElemType t);
  virtual ~GdsElemBase() = default;

  // getter
  const char* get_property(int16_t) const;
  size_t get_property_num() const;
  GdsElemType get_elem_type() const;
  const GdsXY& get_xy() const { return _xy; }
  const std::map<int16_t, std::string>& get_property_map() const;
  GdsELFlags get_flags() const;
  bool is_external() const;
  bool is_template_data() const;
  GdsPlexType get_plex() const;

  // setter
  void set_property(int16_t, const std::string&);
  void add_coord(int32_t x, int32_t y);
  void add_coord(const XYCoordinate& c);
  void set_flags(const GdsELFlags&);
  void set_external(bool);
  void set_template_data(bool);
  void set_plex(GdsPlexType);

  // operator
  GdsElemBase& operator=(const GdsElemBase&);

  // function
  void remove_property(int16_t);
  void reset_base();
  virtual void reset() = 0;

 private:
  // members
  GdsELFlags _flags;
  GdsPlexType _plex;  //
  std::map<int16_t, std::string> _property_map;
  GdsElemType _type;
  GdsXY _xy;
};

class GdsElement : public GdsElemBase
{
 public:
  GdsElement() : GdsElemBase(GdsElemType::kElement) {}
  void reset() override;
};

///////////// inline ////////////

inline GdsElemBase::GdsElemBase(GdsElemType t) : _flags(), _plex(), _property_map(), _type(t), _xy()
{
}

inline const char* GdsElemBase::get_property(int16_t attr) const
{
  auto it = _property_map.find(attr);
  return it != _property_map.end() ? it->second.c_str() : "";
}

inline void GdsElemBase::set_property(int16_t attr, const std::string& v)
{
  _property_map[attr] = v;
}

inline size_t GdsElemBase::get_property_num() const
{
  return _property_map.size();
}

inline GdsElemType GdsElemBase::get_elem_type() const
{
  return _type;
}

inline void GdsElemBase::remove_property(int16_t attr)
{
  _property_map.erase(attr);
}

inline void GdsElemBase::add_coord(int32_t x, int32_t y)
{
  _xy.add_coord(x, y);
}

inline void GdsElemBase::add_coord(const XYCoordinate& c)
{
  _xy.add_coord(c);
}

inline GdsElemBase& GdsElemBase::operator=(const GdsElemBase& rhs)
{
  _property_map = rhs._property_map;
  _type = rhs._type;
  _xy = rhs._xy;

  return *this;
}

inline const std::map<int16_t, std::string>& GdsElemBase::get_property_map() const
{
  return _property_map;
}

inline GdsELFlags GdsElemBase::get_flags() const
{
  return _flags;
}

inline void GdsElemBase::set_flags(const GdsELFlags& f)
{
  _flags = f;
}

inline void GdsElemBase::set_external(bool b)
{
  _flags.set_is_external(b);
}

inline void GdsElemBase::set_template_data(bool b)
{
  _flags.set_is_template(b);
}

inline bool GdsElemBase::is_external() const
{
  return _flags.is_external();
}

inline bool GdsElemBase::is_template_data() const
{
  return _flags.is_template();
}

inline void GdsElemBase::reset_base()
{
  _flags.reset();
  _property_map.clear();
  _xy.clear();
  _flags.reset();
}

// A unique positive number which is common to all elementsof the plex to which this element belongs.
inline GdsPlexType GdsElemBase::get_plex() const
{
  return _plex;
}

inline void GdsElemBase::set_plex(GdsPlexType plex)
{
  _plex = plex;
}

inline void GdsElement::reset()
{
  reset_base();
}

}  // namespace idb