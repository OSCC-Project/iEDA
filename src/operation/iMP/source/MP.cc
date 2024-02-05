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
#include "MP.hh"

#include <functional>

#include "BlkClustering.hh"
#include "HierPlacer.hh"
#include "Logger.hpp"
#include "SAPlacer.hh"
namespace imp {

void MP::runMP()
{
  // BlkClustering2 clustering{.l1_nparts = 10, .l2_n_parts = 20, .level_num = 1}; // two level place
  BlkClustering2 clustering{.l1_nparts = 100, .level_num = 1};  // one level place
  root().preorder_op(clustering);
  auto placer = SAHierPlacer<int32_t>(root());
  placer.hierPlace(true);
  placer.writePlacement(root(), "/home/liuyuezuo/iEDA-master/build/output/placement.txt");
}

}  // namespace imp