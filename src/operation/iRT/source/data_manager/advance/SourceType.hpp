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

#include "Logger.hpp"

namespace irt {

enum class SourceType
{
  kNone,
  kBlockAndPin,
  kAccessPoint,
  kRoutingResult,
  kDRCShape,
  kPatchShape
};

struct GetSourceTypeName
{
  std::string operator()(const SourceType& source_type) const
  {
    std::string source_type_name;
    switch (source_type) {
      case SourceType::kNone:
        source_type_name = "none";
        break;
      case SourceType::kBlockAndPin:
        source_type_name = "block_and_pin";
        break;
      case SourceType::kAccessPoint:
        source_type_name = "access_point";
        break;
      case SourceType::kRoutingResult:
        source_type_name = "routing_result";
        break;
      case SourceType::kDRCShape:
        source_type_name = "drc_shape";
        break;
      case SourceType::kPatchShape:
        source_type_name = "patch_shape";
        break;
      default:
        LOG_INST.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return source_type_name;
  }
};

}  // namespace irt
