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

enum class ERStage
{
  kNone,
  kEgr2D,
  kEgr3D,
  kEdr
};

struct GetERStageName
{
  std::string operator()(const ERStage& violation_type) const
  {
    std::string violation_type_name;
    switch (violation_type) {
      case ERStage::kNone:
        violation_type_name = "none";
        break;
      case ERStage::kEgr2D:
        violation_type_name = "egr2D";
        break;
      case ERStage::kEgr3D:
        violation_type_name = "egr3D";
        break;
      case ERStage::kEdr:
        violation_type_name = "edr";
        break;
      default:
        RTLOG.error(Loc::current(), "Unrecognized type!");
        break;
    }
    return violation_type_name;
  }
};

struct GetERStageByName
{
  ERStage operator()(const std::string& violation_type_name) const
  {
    ERStage violation_type;
    if (violation_type_name == "egr2D") {
      violation_type = ERStage::kEgr2D;
    } else if (violation_type_name == "egr3D") {
      violation_type = ERStage::kEgr3D;
    } else if (violation_type_name == "edr") {
      violation_type = ERStage::kEdr;
    } else {
      violation_type = ERStage::kNone;
    }
    return violation_type;
  }
};

}  // namespace irt
