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
  // for testing, packing vertically
  // int32_t total_width = 0;
  // int32_t max_height = 0;
  double area = 0;
  for (const auto& shape_curve : sub_shape_curves) {
    area += shape_curve.get_area();
  }
  int32_t width = std::sqrt(1.2 * area);
  return {{width, width}};
}

class HierPlacer
{
 public:
  explicit HierPlacer(Block& root_cluster) : _root_cluster(root_cluster) {}
  ~HierPlacer() = default;
  void place(std::function<void(Block&)> place_solver)
  {
    _root_cluster.init_cell_area();                            // init stdcell-area && macro-area
    _root_cluster.coarse_shaping(get_packing_shapes);          // init discrete-shapes bottom-up (coarse-shaping, only considers macros)
    auto place_op = [place_solver](imp::Block& blk) -> void {  // place hier-cluster top-down
      if (!blk.isBlock() || !(blk.is_macro_cluster() || blk.is_mixed_cluster())) {
        return;  // only place cluster with macros..
      }
      // fine-shaping at current level before placement
      blk.clipChildrenShapes();      // clip discrete-shapes larger than parent-clusters bounding-box
      blk.addChildrenStdcellArea();  // add stdcell area
      place_solver(blk);
      //
    };
    // parallel_preorder_op(place_op);
    _root_cluster.preorder_op(place_op);
  }

 private:
  Block& _root_cluster;
};

void MP::runMP()
{
  BlkClustering clustering{5, 20};
  root().parallel_preorder_op(clustering);
  // root().init_cell_area();
  // root().coarse_shaping(get_packing_shapes);
  // root().hierPlace([](Block& blk) {
  //   // for (auto&& i : blk.netlist().vRange()) {
  //   //   auto sub_obj = i.property();

  //   // }
  // });

  auto placer = HierPlacer(root());
  placer.place([](Block& blk) { std::cout << "pretend to place..." << std::endl; });
}

}  // namespace imp
