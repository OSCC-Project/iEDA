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

#include "JsonAref.hpp"
#include "JsonBoundary.hpp"
#include "JsonBox.hpp"
#include "JsonElement.hpp"
#include "JsonNode.hpp"
#include "JsonPath.hpp"
#include "JsonSref.hpp"
#include "JsonText.hpp"
#include "JsonTimestamp.hpp"

namespace idb {

class JsonStruct
{
 public:
  // constructor
  JsonStruct();
  JsonStruct(const JsonStruct&);
  explicit JsonStruct(const std::string&);
  ~JsonStruct();

  // getter
  time_t get_bgn_str() const { return _ts.beg; }
  time_t get_last_mod() const { return _ts.last; }
  std::string get_name() const { return _name; }
  const std::vector<JsonElemBase*>& get_element_list() const { return _element_list; }

  // setter
  void set_name(const std::string& s) { _name = s; }
  void add_element(JsonElemBase*);
  void add_element(const JsonElement&);
  void add_element(const JsonBoundary&);
  void add_element(const JsonPath&);
  void add_element(const JsonSref&);
  void add_element(const JsonAref&);
  void add_element(const JsonText&);
  void add_element(const JsonNode&);
  void add_element(const JsonBox&);

  // operator
  JsonStruct& operator=(const JsonStruct&);

  // function
  void clear();
  void clear_element_list();

 private:
  // members
  std::string _name;  // structure name
  JsonTimestamp _ts;   // structure timestamp
  std::vector<JsonElemBase*> _element_list;
};

}  // namespace idb