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
#include "GdsStruct.hpp"

namespace idb {

GdsStruct::GdsStruct() : _ts(), _name(), _element_list()
{
}

GdsStruct::GdsStruct(const std::string& s) : _ts(), _name(s), _element_list()
{
}

GdsStruct::GdsStruct(const GdsStruct& other)
{
  *this = other;
}

GdsStruct::~GdsStruct()
{
  clear_element_list();
}

void GdsStruct::add_element(GdsElemBase* e)
{
  if (e) {
    _element_list.emplace_back(e);
  }
}

void GdsStruct::clear()
{
  _ts.reset();
  _name.clear();
  clear_element_list();
}

void GdsStruct::clear_element_list()
{
  for (auto e : _element_list) {
    delete e;
  }
  _element_list.clear();
}

void GdsStruct::add_element(const GdsElement& e)
{
  auto cpy = new GdsElement();
  *cpy = e;
  add_element(cpy);
}

void GdsStruct::add_element(const GdsBoundary& e)
{
  auto cpy = new GdsBoundary();
  *cpy = e;
  add_element(cpy);
}

void GdsStruct::add_element(const GdsPath& e)
{
  auto cpy = new GdsPath();
  *cpy = e;
  add_element(cpy);
}

void GdsStruct::add_element(const GdsSref& e)
{
  auto cpy = new GdsSref();
  *cpy = e;
  add_element(cpy);
}

void GdsStruct::add_element(const GdsAref& e)
{
  auto cpy = new GdsAref();
  *cpy = e;
  add_element(cpy);
}

void GdsStruct::add_element(const GdsText& e)
{
  auto cpy = new GdsText();
  *cpy = e;
  add_element(cpy);
}

void GdsStruct::add_element(const GdsNode& e)
{
  auto cpy = new GdsNode();
  *cpy = e;
  add_element(cpy);
}

void GdsStruct::add_element(const GdsBox& e)
{
  auto cpy = new GdsBox();
  *cpy = e;
  add_element(cpy);
}

GdsStruct& GdsStruct::operator=(const GdsStruct& rhs)
{
  _ts = rhs._ts;
  _name = rhs._name;

  clear_element_list();
  for (auto e : rhs._element_list) {
    switch (e->get_elem_type()) {
      case GdsElemType::kElement:
        add_element(*dynamic_cast<GdsElement*>(e));
        break;
      case GdsElemType::kBoundary:
        add_element(*dynamic_cast<GdsBoundary*>(e));
        break;
      case GdsElemType::kPath:
        add_element(*dynamic_cast<GdsPath*>(e));
        break;
      case GdsElemType::kSref:
        add_element(*dynamic_cast<GdsSref*>(e));
        break;
      case GdsElemType::kAref:
        add_element(*dynamic_cast<GdsAref*>(e));
        break;
      case GdsElemType::kText:
        add_element(*dynamic_cast<GdsText*>(e));
        break;
      case GdsElemType::kNode:
        add_element(*dynamic_cast<GdsNode*>(e));
        break;
      case GdsElemType::kBox:
        add_element(*dynamic_cast<GdsBox*>(e));
        break;

      default:
        break;
    }  // end switch
  }    // end for

  return *this;
}

}  // namespace idb
