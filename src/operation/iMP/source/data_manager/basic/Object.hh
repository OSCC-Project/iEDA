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

 private:
  friend class Block;
  friend class Instance;
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