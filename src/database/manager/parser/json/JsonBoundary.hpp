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

#include "JsonElement.hpp"

namespace idb {

class JsonBoundary : public JsonElemBase
{
 public:
  JsonBoundary() : JsonElemBase(JsonElemType::kBoundary), layer(0), data_type(0) {}

  JsonBoundary& operator=(const JsonBoundary& rhs)
  {
    JsonElemBase::operator=(rhs);
    layer = rhs.layer;
    data_type = rhs.data_type;

    return *this;
  }

  void reset() override
  {
    reset_base();
    layer = 0;
    data_type = 0;
  }

  // members
  JsonLayer layer;
  JsonDataType data_type;
};

}  // namespace idb
