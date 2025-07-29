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

#include "init_sta.hh"
#include "vec_api.h"
#include "vectorization.h"

namespace ivec {

bool VectorizationApi::buildVectorizationLayoutData(const std::string path)
{
  Vectorization vectorization;

  vectorization.buildLayoutData(path);

  return true;
}

bool VectorizationApi::buildVectorizationGraphData(const std::string path)
{
  Vectorization vectorization;

  vectorization.buildGraphData(path);

  return true;
}

std::map<int, VecNet> VectorizationApi::getGraph(std::string path)
{
  Vectorization vectorization;

  return vectorization.getGraph(path);
}

bool VectorizationApi::buildVectorizationFeature(const std::string dir)
{
  Vectorization vectorization;

  vectorization.buildFeature(dir);

  return true;
}

bool VectorizationApi::runVecSTA(const std::string dir)
{
  Vectorization vectorization;

  vectorization.buildLayoutData(dir);
  vectorization.buildGraphDataWithoutSave(dir);

  vectorization.runVecSTA(dir);

  return true;
}
}  // namespace ivec