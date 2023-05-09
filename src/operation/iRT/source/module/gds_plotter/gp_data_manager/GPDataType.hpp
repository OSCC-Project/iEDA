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
  kNone = 0,
  kText = 1,
  kBoundingBox = 2,
  kPort = 3,
  kAccessPoint = 4,
  kGuide = 5,
  kPreferTrack = 6,
  kNonpreferTrack = 7,
  kWire = 8,
  kEnclosure = 9,
  kCut = 10,
  kBlockage = 11,
  kConnection = 12
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
      case GPDataType::kText:
        data_type_name = "text";
        break;
      case GPDataType::kBoundingBox:
        data_type_name = "bounding_box";
        break;
      case GPDataType::kPort:
        data_type_name = "port";
        break;
      case GPDataType::kAccessPoint:
        data_type_name = "access_point";
        break;
      case GPDataType::kGuide:
        data_type_name = "guide";
        break;
      case GPDataType::kPreferTrack:
        data_type_name = "prefer_track";
        break;
      case GPDataType::kNonpreferTrack:
        data_type_name = "nonprefer_track";
        break;
      case GPDataType::kWire:
        data_type_name = "wire";
        break;
      case GPDataType::kEnclosure:
        data_type_name = "enclosure";
        break;
      case GPDataType::kCut:
        data_type_name = "cut";
        break;
      case GPDataType::kBlockage:
        data_type_name = "blockage";
        break;
      case GPDataType::kConnection:
        data_type_name = "connection";
        break;
      default:
        LOG_INST.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return data_type_name;
  }
};

}  // namespace irt
