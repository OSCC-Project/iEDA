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
/**
 * @project		iplf
 * @file		FileHeader.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Process file
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

using std::string;
using std::vector;

namespace iplf {
enum class FileModuleId : int32_t
{
  kNone = 0,
  kPA = 1,
  kCTS = 2,
  kDRC = 3,
  kPL = 4,
  kMax
};
enum class PaDbId : int32_t
{
  kNone = 0,
  kPaData = 1,
  kMax
};
enum class CtsDbId : int32_t
{
  kNone = 0,
  kCtsRoutingData = 1,
  kCtsGuiData = 2,
  kMax
};

enum class PlDbId : int32_t
{
  kNone = 0,
  kPlInstanceData = 1,
  kMax
};

enum class DrcDbId : int32_t
{
  kNone = 0,
  kDrcDetailInfo,
  kCutEOL,
  kCutSpacing,
  kCutEnclosure,
  kMetalEOL,
  kMetalShort,
  kMetalPRL,
  kMetalNotch,
  kMinStep,
  kMinArea,
  kMax
};

struct FileHeader
{
  int32_t _module_id;
  int32_t _object_id;
  uint64_t _data_size;
};

}  // namespace iplf
