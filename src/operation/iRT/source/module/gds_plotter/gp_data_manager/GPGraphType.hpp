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

enum class GPGraphType
{
  kNone,
  kOpen,
  kClose,
  kInfo,
  kNeighbor,
  kKey,
  kTrackAxis,
  kPath,
  kBlockAndPin,
  kEnclosure,
  kOtherPanel,
  kSelfPanel,
  kKnownPanel,
  kOtherBox,
  kSelfBox
};

struct GetGPGraphTypeName
{
  std::string operator()(const GPGraphType& data_type) const
  {
    std::string data_type_name;
    switch (data_type) {
      case GPGraphType::kNone:
        data_type_name = "none";
        break;
      case GPGraphType::kOpen:
        data_type_name = "open";
        break;
      case GPGraphType::kClose:
        data_type_name = "close";
        break;
      case GPGraphType::kInfo:
        data_type_name = "info";
        break;
      case GPGraphType::kNeighbor:
        data_type_name = "neighbor";
        break;
      case GPGraphType::kKey:
        data_type_name = "key";
        break;
      case GPGraphType::kTrackAxis:
        data_type_name = "scale_axis";
        break;
      case GPGraphType::kPath:
        data_type_name = "path";
        break;
      case GPGraphType::kBlockAndPin:
        data_type_name = "block_pin";
        break;
      case GPGraphType::kEnclosure:
        data_type_name = "enclosure";
        break;
      case GPGraphType::kOtherPanel:
        data_type_name = "other_panel";
        break;
      case GPGraphType::kSelfPanel:
        data_type_name = "self_panel";
        break;
      case GPGraphType::kKnownPanel:
        data_type_name = "known_panel";
        break;
      case GPGraphType::kOtherBox:
        data_type_name = "other_box";
        break;
      case GPGraphType::kSelfBox:
        data_type_name = "self_box";
        break;
      default:
        LOG_INST.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return data_type_name;
  }
};

}  // namespace irt
