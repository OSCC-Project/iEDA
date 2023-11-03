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

#include <string>

#include "JsonElement.hpp"
#include "JsonStrans.hpp"

namespace idb {

enum class JsonPresentation
{
  kLeft = 0b00,
  kCenterH = 0b01,
  kRight = 0b10,
  kDefaultH = kLeft,

  kBottom = 0b1000,
  kCenterV = 0b0100,
  kTop = 0b0000,
  kDefaultV = kBottom,

  kDefault = kDefaultV | kDefaultH,

  kCenter = kCenterV | kCenterH,
  kBottomLeft = kBottom | kLeft,
  kBottomRight = kBottom | kRight,
  kTopLeft = kTop | kLeft,
  kTopRight = kTop | kRight,
};

class JsonText : public JsonElemBase
{
 public:
  JsonText()
      : JsonElemBase(JsonElemType::kText),
        layer(0),
        text_type(0),
        presentation(JsonPresentation::kDefault),
        path_type(JsonPathType::kDefault),
        width(0),
        strans(),
        str()
  {
  }

  JsonText& operator=(const JsonText& rhs)
  {
    JsonElemBase::operator=(rhs);
    layer = rhs.layer;
    text_type = rhs.text_type;
    presentation = rhs.presentation;
    path_type = rhs.path_type;
    width = rhs.width;
    strans = rhs.strans;
    str = rhs.str;

    return *this;
  }

  void reset() override
  {
    reset_base();
    layer = 0;
    text_type = 0;
    presentation = JsonPresentation::kDefault;
    path_type = JsonPathType::kDefault;
    width = 0;
    strans.reset();
    str.clear();
  }

  // members
  JsonLayer layer;
  JsonTextType text_type;
  JsonPresentation presentation;
  JsonPathType path_type;
  JsonWidth width;
  JsonStrans strans;
  std::string str;
};

}  // namespace idb
