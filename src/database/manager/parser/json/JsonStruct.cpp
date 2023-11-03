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
#include "JsonStruct.hpp"

namespace idb {

JsonStruct::JsonStruct() : _ts(), _name(), _element_list()
{
}

JsonStruct::JsonStruct(const std::string& s) : _ts(), _name(s), _element_list()
{
}

JsonStruct::JsonStruct(const JsonStruct& other)
{
  *this = other;
}

JsonStruct::~JsonStruct()
{
  clear_element_list();
}

void JsonStruct::add_element(JsonElemBase* e)
{
  if (e) {
    _element_list.emplace_back(e);
  }
}

void JsonStruct::clear()
{
  _ts.reset();
  _name.clear();
  clear_element_list();
}

void JsonStruct::clear_element_list()
{
  for (auto e : _element_list) {
    delete e;
  }
  _element_list.clear();
}

void JsonStruct::add_element(const JsonElement& e)
{
  auto cpy = new JsonElement();
  *cpy = e;
  add_element(cpy);
}

void JsonStruct::add_element(const JsonBoundary& e)
{
  auto cpy = new JsonBoundary();
  *cpy = e;
  add_element(cpy);
}

void JsonStruct::add_element(const JsonPath& e)
{
  auto cpy = new JsonPath();
  *cpy = e;
  add_element(cpy);
}

void JsonStruct::add_element(const JsonSref& e)
{
  auto cpy = new JsonSref();
  *cpy = e;
  add_element(cpy);
}

void JsonStruct::add_element(const JsonAref& e)
{
  auto cpy = new JsonAref();
  *cpy = e;
  add_element(cpy);
}

void JsonStruct::add_element(const JsonText& e)
{
  auto cpy = new JsonText();
  *cpy = e;
  add_element(cpy);
}

void JsonStruct::add_element(const JsonNode& e)
{
  auto cpy = new JsonNode();
  *cpy = e;
  add_element(cpy);
}

void JsonStruct::add_element(const JsonBox& e)
{
  auto cpy = new JsonBox();
  *cpy = e;
  add_element(cpy);
}

JsonStruct& JsonStruct::operator=(const JsonStruct& rhs)
{
  _ts = rhs._ts;
  _name = rhs._name;

  clear_element_list();
  for (auto e : rhs._element_list) {
    switch (e->get_elem_type()) {
      case JsonElemType::kElement:
        add_element(*dynamic_cast<JsonElement*>(e));
        break;
      case JsonElemType::kBoundary:
        add_element(*dynamic_cast<JsonBoundary*>(e));
        break;
      case JsonElemType::kPath:
        add_element(*dynamic_cast<JsonPath*>(e));
        break;
      case JsonElemType::kSref:
        add_element(*dynamic_cast<JsonSref*>(e));
        break;
      case JsonElemType::kAref:
        add_element(*dynamic_cast<JsonAref*>(e));
        break;
      case JsonElemType::kText:
        add_element(*dynamic_cast<JsonText*>(e));
        break;
      case JsonElemType::kNode:
        add_element(*dynamic_cast<JsonNode*>(e));
        break;
      case JsonElemType::kBox:
        add_element(*dynamic_cast<JsonBox*>(e));
        break;

      default:
        break;
    }  // end switch
  }    // end for

  return *this;
}

}  // namespace idb
