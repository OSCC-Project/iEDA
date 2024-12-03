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

#include "lm_api.h"

#include "large_model.h"

namespace ilm {

bool LargeModelApi::buildLargeModelLayoutData(const std::string path)
{
  LargeModel large_model;

  large_model.buildLayoutData(path);

  return true;
}

bool LargeModelApi::buildLargeModelGraphData(const std::string path)
{
  LargeModel large_model;

  large_model.buildGraphData(path);

  return true;
}

std::map<int, LmNet> LargeModelApi::getGraph(std::string path)
{
  LargeModel large_model;

  return large_model.getGraph(path);
}

bool LargeModelApi::buildLargeModelFeature(const std::string path)
{
  //   LargeModel large_model;

  //   large_model.buildLayoutData(path);

  return true;
}
}  // namespace ilm