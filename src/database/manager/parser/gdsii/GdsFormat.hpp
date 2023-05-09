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

#include <assert.h>

#include <string>

namespace idb {

enum class GdsFormatType
{
  kGDSII_Archive = 0b01,
  kGDSII_Filtered = 0b10,
  kEDSM_Archive = 0b11,
  kEDSHI_Filtered = 0b100,
};

class GdsFormat
{
 public:
  GdsFormat() : type(GdsFormatType::kGDSII_Archive), mask() {}

  bool is_archive() const;
  bool is_filtered() const;

  GdsFormatType type;
  std::string mask;
};

//////////// inline ///////

inline bool GdsFormat::is_archive() const
{
  return (int) type & 0b1;
}

inline bool GdsFormat::is_filtered() const
{
  return !is_archive();
}

}  // namespace idb