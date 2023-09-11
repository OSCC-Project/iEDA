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

enum class GPLayoutType
{
  kNone,
  kText,
  kPinShape,
  kAccessPoint,
  kGuide,
  kPreferTrack,
  kNonpreferTrack,
  kWire,
  kEnclosure,
  kCut,
  kPatch,
  kBlockage
};

struct GetGPLayoutTypeName
{
  std::string operator()(const GPLayoutType& data_type) const
  {
    std::string data_type_name;
    switch (data_type) {
      case GPLayoutType::kNone:
        data_type_name = "none";
        break;
      case GPLayoutType::kText:
        data_type_name = "text";
        break;
      case GPLayoutType::kPinShape:
        data_type_name = "pin_shape";
        break;
      case GPLayoutType::kAccessPoint:
        data_type_name = "access_point";
        break;
      case GPLayoutType::kGuide:
        data_type_name = "guide";
        break;
      case GPLayoutType::kPreferTrack:
        data_type_name = "prefer_track";
        break;
      case GPLayoutType::kNonpreferTrack:
        data_type_name = "nonprefer_track";
        break;
      case GPLayoutType::kWire:
        data_type_name = "wire";
        break;
      case GPLayoutType::kEnclosure:
        data_type_name = "enclosure";
        break;
      case GPLayoutType::kCut:
        data_type_name = "cut";
        break;
      case GPLayoutType::kPatch:
        data_type_name = "patch";
        break;
      case GPLayoutType::kBlockage:
        data_type_name = "blockage";
        break;
      default:
        LOG_INST.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return data_type_name;
  }
};

}  // namespace irt
