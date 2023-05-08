#pragma once

#include "Orientation.hpp"
#include "RTU.hpp"

namespace irt {

class DRScaleOrient
{
 public:
  DRScaleOrient() = default;
  explicit DRScaleOrient(const irt_int scale) { _scale = scale; }
  DRScaleOrient(const irt_int scale, const Orientation orientation)
  {
    _scale = scale;
    _orientation_set.insert(orientation);
  }
  ~DRScaleOrient() = default;
  // getter
  irt_int get_scale() const { return _scale; }
  std::set<Orientation>& get_orientation_set() { return _orientation_set; }
  // setter
  void set_scale(const irt_int scale) { _scale = scale; }
  void set_orientation_set(const std::set<Orientation>& orientation_set) { _orientation_set = orientation_set; }
  // function

 private:
  irt_int _scale = -1;
  std::set<Orientation> _orientation_set;
};

}  // namespace irt
