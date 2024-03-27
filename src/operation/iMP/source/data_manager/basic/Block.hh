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

#ifndef IMP_BLOCK_H
#define IMP_BLOCK_H
#include <functional>
#include <future>
#include <ranges>
#include <stdexcept>

#include "Instance.hh"
#include "Netlist.hh"
#include "Object.hh"
#include "ShapeCurve.hh"

namespace imp {
class Block;

template <typename Operator>
concept BlockOperator = requires { std::is_invocable_v<Operator, Block>; };

class Block : public Object, public std::enable_shared_from_this<Block>
{
 public:
  Block() = default;
  Block(std::string name, std::shared_ptr<Netlist> netlist = nullptr, std::shared_ptr<Object> parent = nullptr);
  ~Block() = default;

  virtual OBJ_TYPE object_type() const override { return OBJ_TYPE::kBlock; }
  virtual geo::box<int32_t> boundingbox() const override { return Object::transform(get_curr_shape()); }
  // void set_shape(const geo::box<int32_t>& box)
  // {
  //   set_min_corner(geo::lx(box), geo::ly(box));
  //   _shape = geo::make_box(0, 0, geo::width(box), geo::height(box));
  // }

  size_t level() const;
  bool is_leaf() const;
  Netlist& netlist();
  const Netlist& netlist() const;
  void set_netlist(std::shared_ptr<Netlist> netlist);

  void set_shape_curve(const ShapeCurve<int32_t>& shape_curve) { _shape_curve = shape_curve; }
  void set_shape_curve(const geo::box<int32_t>& box)
  {
    // only set a box shape (no changeable shape curve)
    set_shape_curve({{geo::width(box), geo::height(box)}}, 0, false);
  }
  void set_shape_curve(const std::vector<std::pair<int32_t, int32_t>>& discrete_shapes, float continuous_shapes_area, bool use_clip = true)
  {
    // set shape with shapeCurve
    _shape_curve.setShapes(discrete_shapes, continuous_shapes_area, use_clip);
  }
  const ShapeCurve<int32_t>& get_shape_curve() const { return _shape_curve; }
  ShapeCurve<int32_t>& get_shape_curve() { return _shape_curve; }
  geo::box<int32_t> get_curr_shape() const { return geo::make_box(0, 0, _shape_curve.get_width(), _shape_curve.get_height()); }
  void set_macro_num(size_t macro_num) { _macro_num = macro_num; }
  void set_macro_area(float macro_area) { _macro_area = macro_area; }
  void set_stdcell_area(float stdcell_area) { _stdcell_area = stdcell_area; }
  void set_io_area(float io_area) { _io_area = io_area; }
  void set_fixed() { _is_fixed = true; }
  void set_unfixed() { _is_fixed = false; }
  void add_blockage(int32_t lx, int32_t ly, int32_t ux, int32_t uy) { _blockages.push_back(geo::make_box(lx, ly, ux, uy)); }
  const std::vector<geo::box<int32_t>>& get_blockages() const { return _blockages; }
  size_t get_macro_num() const { return _macro_num; }
  float get_macro_area() const { return _macro_area; }
  float get_stdcell_area() const { return _stdcell_area; }
  float get_io_area() const { return _io_area; }
  bool isFixed() const { return _is_fixed; }
  bool is_macro_cluster() { return _macro_area > 0 && _stdcell_area <= 0 && _io_area <= 0; }
  bool is_stdcell_cluster() { return _macro_area <= 0 && _stdcell_area > 0 && _io_area <= 0; }
  bool is_mixed_cluster() { return _macro_area > 0 && _stdcell_area > 0 && _io_area <= 0; }
  bool is_io_cluster() { return _io_area >= 0 && _macro_area <= 0 && _stdcell_area <= 0; }

  template <BlockOperator Operator>
  void preorder_op(Operator op);
  /**
   * @brief Parallel preorder
   *
   * @tparam Operator
   * @param op
   */
  template <BlockOperator Operator>
  void parallel_preorder_op(Operator op);

  template <BlockOperator Operator>
  void postorder_op(Operator op);

  template <BlockOperator Operator>
  void level_order_op(Operator op);
  using std::enable_shared_from_this<Block>::shared_from_this;

 private:
  // geo::box<int32_t> _shape;
  std::shared_ptr<Netlist> _netlist;
  std::vector<geo::box<int32_t>> _blockages;
  size_t _macro_num = 0;
  float _macro_area = 0;
  float _stdcell_area = 0;
  float _io_area = 0;
  bool _is_fixed = false;
  ShapeCurve<int32_t> _shape_curve;  // containing stdcell area
};

template <BlockOperator Operator>
inline void Block::preorder_op(Operator op)
{
  op(*this);
  for (auto&& i : _netlist->vRange()) {
    auto obj = i.property();
    if (!obj->isBlock())
      continue;
    auto sub_block = std::static_pointer_cast<Block, Object>(obj);
    sub_block->preorder_op(op);
  }
}

template <BlockOperator Operator>
inline void Block::postorder_op(Operator op)
{
  for (auto&& i : _netlist->vRange()) {
    auto obj = i.property();
    if (!obj->isBlock())
      continue;
    auto sub_block = std::static_pointer_cast<Block, Object>(obj);
    sub_block->postorder_op(op);
  }
  op(*this);
}

template <BlockOperator Operator>
void Block::parallel_preorder_op(Operator op)
{
  op(*this);

  std::vector<std::future<void>> futures;
  for (auto&& i : _netlist->vRange()) {
    auto obj = i.property();
    if (!obj->isBlock())
      continue;

    auto sub_block = std::static_pointer_cast<Block, Object>(obj);
    futures.push_back(std::async(std::launch::async, &Block::preorder_op<Operator>, sub_block, op));
  }

  for (auto&& future : futures) {
    future.get();
  }
}
}  // namespace imp

#endif