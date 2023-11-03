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

namespace idb {

using JsonHeader = int16_t;
using JsonGenerations = int16_t;
using JsonLayer = int16_t;
using JsonDataType = int16_t;
using JsonWidth = int16_t;
using JsonTextType = int16_t;
using JsonNodeType = int16_t;
using JsonBoxType = int16_t;
using JsonPlexType = int16_t;

enum class JsonPathType
{
  kSquareEnd = 0,
  kRoundEnd = 1,
  kExtendSquareEnd = 2,

  kDefault = kSquareEnd
};

}  // namespace idb
