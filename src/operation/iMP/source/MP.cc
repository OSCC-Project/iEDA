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
namespace imp {

std::vector<std::pair<int32_t, int32_t>> get_packing_shapes(std::vector<ShapeCurve<int32_t>>& sub_shape_curves)
{
  // packing vertically
  int32_t total_width = 0;
  int32_t max_height = 0;
  for (const auto& shape_curve : sub_shape_curves) {
    total_width += shape_curve.get_width();
    max_height = std::max(max_height, shape_curve.get_height());
  }
  return {{total_width, max_height}};
}

void MP::runMP()
{
  BlkClustering clustering{5, 20};
  root().parallel_preorder_op(clustering);
  root().init_cell_area();
  root().coarse_shaping(get_packing_shapes);
}

}  // namespace imp
