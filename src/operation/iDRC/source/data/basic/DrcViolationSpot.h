#pragma once

#include <vector>

#include "DrcEnum.h"
#include "DrcRect.h"
#include "DrcRectangle.h"
namespace idrc {
class DrcViolationSpot
{
 public:
  DrcViolationSpot() = default;
  /// getter
  ViolationType get_violation_type() { return _violation_type; }
  std::string get_layer() { return _layer_name; }
  int get_layer_id() { return _layer_id; }
  int get_net_id() { return _net_id; }
  int get_min_x() { return std::min(_vio_box_min_corner.get_x(), _vio_box_max_corner.get_x()); }
  int get_min_y() { return std::min(_vio_box_min_corner.get_y(), _vio_box_max_corner.get_y()); }
  int get_max_x() { return std::max(_vio_box_min_corner.get_x(), _vio_box_max_corner.get_x()); }
  int get_max_y() { return std::max(_vio_box_min_corner.get_y(), _vio_box_max_corner.get_y()); }

  void set_vio_type(ViolationType type) { _violation_type = type; }
  void set_layer_name(const std::string& str) { _layer_name = str; }
  void set_layer_id(const int& layer_id) { _layer_id = layer_id; }
  void set_net_id(const int& net_id) { _net_id = net_id; }
  void setCoordinate(int lb_x, int lb_y, int rt_x, int rt_y)
  {
    _vio_box_min_corner.set_x(lb_x);
    _vio_box_min_corner.set_y(lb_y);
    _vio_box_max_corner.set_x(rt_x);
    _vio_box_max_corner.set_y(rt_y);
  }

  private:
  ViolationType _violation_type;
  std::string _layer_name;
  int _layer_id;
  int _net_id;
  DrcCoordinate<int> _vio_box_min_corner;
  DrcCoordinate<int> _vio_box_max_corner;
};
}  // namespace idrc