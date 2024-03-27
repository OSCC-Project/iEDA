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

#include "engine_geometry_creator.h"

#include "engine_geometry.h"
#include "geometry_boost.h"

namespace ieda_solver {

EngineGeometry* EngineGeometryCreator::create(EngineGeometryType type)
{
  EngineGeometry* geo = nullptr;

  switch (type) {
    case EngineGeometryType::kBoost:
      geo = new GeometryBoost();
      break;
    case EngineGeometryType::kCustomized:

      break;

    default:
      break;
  }

  return geo;
}

}  // namespace ieda_solver