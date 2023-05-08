#pragma once

#include <stdint.h>

namespace idb {

using GdsHeader = int16_t;
using GdsGenerations = int16_t;
using GdsLayer = int16_t;
using GdsDataType = int16_t;
using GdsWidth = int16_t;
using GdsTextType = int16_t;
using GdsNodeType = int16_t;
using GdsBoxType = int16_t;
using GdsPlexType = int16_t;

enum class GdsPathType
{
  kSquareEnd = 0,
  kRoundEnd = 1,
  kExtendSquareEnd = 2,

  kDefault = kSquareEnd
};

}  // namespace idb
