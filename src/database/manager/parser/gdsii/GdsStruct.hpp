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

#include <time.h>

#include <string>
#include <vector>

#include "GdsAref.hpp"
#include "GdsBoundary.hpp"
#include "GdsBox.hpp"
#include "GdsElement.hpp"
#include "GdsNode.hpp"
#include "GdsPath.hpp"
#include "GdsSref.hpp"
#include "GdsText.hpp"
#include "GdsTimestamp.hpp"

namespace idb {

class GdsStruct
{
 public:
  // constructor
  GdsStruct();
  GdsStruct(const GdsStruct&);
  explicit GdsStruct(const std::string&);
  ~GdsStruct();

  // getter
  time_t get_bgn_str() const { return _ts.beg; }
  time_t get_last_mod() const { return _ts.last; }
  std::string get_name() const { return _name; }
  const std::vector<GdsElemBase*>& get_element_list() const { return _element_list; }

  // setter
  void set_name(const std::string& s) { _name = s; }
  void add_element(GdsElemBase*);
  void add_element(const GdsElement&);
  void add_element(const GdsBoundary&);
  void add_element(const GdsPath&);
  void add_element(const GdsSref&);
  void add_element(const GdsAref&);
  void add_element(const GdsText&);
  void add_element(const GdsNode&);
  void add_element(const GdsBox&);

  // operator
  GdsStruct& operator=(const GdsStruct&);

  // function
  void clear();
  void clear_element_list();

 private:
  // members
  GdsTimestamp _ts;   // structure timestamp
  std::string _name;  // structure name
  std::vector<GdsElemBase*> _element_list;
};

}  // namespace idb