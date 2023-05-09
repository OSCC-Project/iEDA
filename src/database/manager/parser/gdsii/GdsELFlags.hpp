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

namespace idb {

// external library flag
class GdsELFlags
{
 public:
  struct FlagsBits
  {
    unsigned unused : 14;
    unsigned is_external : 1;
    unsigned is_template : 1;
  };

  union FlagsValue
  {
    uint16_t value = 0;
    FlagsBits bits;
  };

  // constructor
  GdsELFlags() : flag() {}

  // getter
  bool is_external() const;
  bool is_template() const;
  auto get_value() const;

  // setter
  void set_is_external(bool);
  void set_is_template(bool);

  // function
  void reset();

  // members
  FlagsValue flag;
};

//////// inline //////

inline bool GdsELFlags::is_external() const
{
  return flag.bits.is_external;
}

inline bool GdsELFlags::is_template() const
{
  return flag.bits.is_template;
}

inline void GdsELFlags::set_is_external(bool v)
{
  flag.bits.is_external = v;
}

inline void GdsELFlags::set_is_template(bool v)
{
  flag.bits.is_template = v;
}

inline void GdsELFlags::reset()
{
  flag.value = 0;
}

inline auto GdsELFlags::get_value() const
{
  return flag.value;
}

}  // namespace idb
