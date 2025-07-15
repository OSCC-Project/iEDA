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

#include "Logger.hpp"

namespace irt {

enum class GPDataType
{
  kNone,
  kOpen,
  kClose,
  kInfo,
  kNeighbor,
  kShadow,
  kKey,
  kPath,
  kPatch,
  kShape,
  kAccessPoint,
  kBestCoord,
  kAxis,
  kOverflow,
  kRouteViolation,
  kPatchViolation
};

struct GetGPDataTypeName
{
  std::string operator()(const GPDataType& data_type) const
  {
    std::string data_type_name;
    switch (data_type) {
      case GPDataType::kNone:
        data_type_name = "none";
        break;
      case GPDataType::kOpen:
        data_type_name = "open";
        break;
      case GPDataType::kClose:
        data_type_name = "close";
        break;
      case GPDataType::kInfo:
        data_type_name = "info";
        break;
      case GPDataType::kNeighbor:
        data_type_name = "neighbor";
        break;
      case GPDataType::kShadow:
        data_type_name = "shadow";
        break;
      case GPDataType::kKey:
        data_type_name = "key";
        break;
      case GPDataType::kPath:
        data_type_name = "path";
        break;
      case GPDataType::kPatch:
        data_type_name = "patch";
        break;
      case GPDataType::kShape:
        data_type_name = "shape";
        break;
      case GPDataType::kAccessPoint:
        data_type_name = "access_point";
        break;
      case GPDataType::kAxis:
        data_type_name = "axis";
        break;
      case GPDataType::kOverflow:
        data_type_name = "overflow";
        break;
      case GPDataType::kRouteViolation:
        data_type_name = "route_violation";
        break;
      case GPDataType::kPatchViolation:
        data_type_name = "patch_violation";
        break;
      default:
        RTLOG.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return data_type_name;
  }
};

}  // namespace irt
