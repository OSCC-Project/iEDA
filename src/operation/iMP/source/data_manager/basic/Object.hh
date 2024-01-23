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

#ifndef IMP_OBJECT_H
#define IMP_OBJECT_H
#include <memory>

#include "Geometry.hh"
#include "Orient.hh"
#include "ShapeCurve.hh"
namespace imp {
enum class OBJ_TYPE
{
  kBlock,
  kInstance
};
class Object
{
 public:
  Object() = default;
  Object(std::string name, std::shared_ptr<Object> parent = nullptr);
  ~Object() = default;
  virtual OBJ_TYPE object_type() const = 0;
  virtual geo::box<int32_t> boundingbox() const = 0;
  bool isBlock() { return object_type() == OBJ_TYPE::kBlock; }
  bool isInstance() { return object_type() == OBJ_TYPE::kInstance; }
  bool isRoot() { return _parent.lock() ? false : true; }
  // virtual Polygon<int32_t> shape() const = 0
  template <typename Geometry>
  Geometry transform(const Geometry& shape) const
  {
    return geo::transform(shape, 1, 1, 0, _min_corner.x(), _min_corner.y());
  }
  void set_min_corner(int32_t lx, int32_t ly)
  {
    _min_corner.set<0>(lx);
    _min_corner.set<1>(ly);
  }
  void set_min_corner(const geo::point<int32_t>& point) { _min_corner = point; }
  void set_orient(Orient orient) { _orient = orient; }

  std::string get_name() const { return _name; }
  geo::point<int32_t> get_min_corner() { return _min_corner; }
  Orient get_orient() { return _orient; }
  Object& parent();
  const Object& parent() const;
  void set_parent(std::shared_ptr<Object> parent);

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
  geo::box<int32_t> get_curr_shape() const { return geo::make_box(0, 0, _shape_curve.get_width(), _shape_curve.get_height()); }
  void set_macro_area(double macro_area) { _macro_area = macro_area; }
  void set_stdcell_area(double stdcell_area) { _stdcell_area = stdcell_area; }
  double get_macro_area() const { return _macro_area; }
  double get_stdcell_area() const { return _stdcell_area; }
  bool is_macro_cluster() { return _macro_area > 0 && _stdcell_area <= 0; }
  bool is_stdcell_cluster() { return _macro_area <= 0 && _stdcell_area > 0; }
  bool is_mixed_cluster() { return _macro_area > 0 && _stdcell_area > 0; }

 private:
  friend class Block;
  friend class Instance;
  double _macro_area = 0;
  double _stdcell_area = 0;
  ShapeCurve<int32_t> _shape_curve;  // containing stdcell area
  geo::point<int32_t> _min_corner;
  Orient _orient;
  std::string _name;
  std::weak_ptr<Object> _parent;
};
inline Object::Object(std::string name, std::shared_ptr<Object> parent) : _name(name), _parent(parent)
{
}

inline Object& Object::parent()
{
  return *_parent.lock();
}

inline const Object& Object::parent() const
{
  return *_parent.lock();
}

inline void Object::set_parent(std::shared_ptr<Object> parent)
{
  _parent = parent;
}
}  // namespace imp

#endif