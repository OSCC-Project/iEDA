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
#ifndef SRC_EVALUATOR_SOURCE_WIRELENGTH_DATABASE_WLFACTORY_HPP_
#define SRC_EVALUATOR_SOURCE_WIRELENGTH_DATABASE_WLFACTORY_HPP_

#include "WL.hpp"
#include "magic_enum.hpp"

namespace eval {

enum WL_TYPE
{
  kWLM,
  kHPWL,
  kHTree,
  kVTree,
  kClique,
  kStar,
  kB2B,
  kFlute,
  kPlaneRoute,
  kSpaceRoute,
  kDR
};

class WLFactory
{
 public:
  WL* createWL(const std::string& wl_type)
  {
    auto enum_type = magic_enum::enum_cast<WL_TYPE>(wl_type);
    switch (enum_type.value()) {
      case kWLM:
        return new WLMWL();
        break;
      case kHPWL:
        return new HPWLWL();
        break;
      case kHTree:
        return new HTreeWL();
        break;
      case kVTree:
        return new VTreeWL();
        break;
      case kClique:
        return new CliqueWL();
        break;
      case kStar:
        return new StarWL();
        break;
      case kB2B:
        return new B2BWL();
        break;
      case kFlute:
        return new FluteWL();
        break;
      case kPlaneRoute:
        return new PlaneRouteWL();
        break;
      case kSpaceRoute:
        return new SpaceRouteWL();
        break;
      case kDR:
        return new DRWL();
        break;
      default:
        return nullptr;
        break;
    }
  }

  WL* createWL(WIRELENGTH_TYPE type)
  {
    switch (type) {
      case (WIRELENGTH_TYPE::kHPWL):
        return new HPWLWL();
        break;
      case (WIRELENGTH_TYPE::kFLUTE):
        return new FluteWL();
        break;
      case (WIRELENGTH_TYPE::kB2B):
        return new B2BWL();
        break;
      default:
        return nullptr;
        break;
    }
  }
};
}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_WIRELENGTH_DATABASE_WLFACTORY_HPP_
