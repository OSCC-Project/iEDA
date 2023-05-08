#pragma once

#include "AccessPoint.hpp"
#include "EXTLayerRect.hpp"
#include "PlanarCoord.hpp"
#include "RTU.hpp"

namespace irt {

class Pin
{
 public:
  Pin() = default;
  ~Pin() = default;
  // getter
  irt_int get_pin_idx() const { return _pin_idx; }
  std::string& get_pin_name() { return _pin_name; }
  std::vector<EXTLayerRect>& get_routing_shape_list() { return _routing_shape_list; }
  std::vector<EXTLayerRect>& get_cut_shape_list() { return _cut_shape_list; }
  std::vector<AccessPoint>& get_access_point_list() { return _access_point_list; }
  // setter
  void set_pin_idx(const irt_int pin_idx) { _pin_idx = pin_idx; }
  void set_pin_name(const std::string& pin_name) { _pin_name = pin_name; }
  void set_routing_shape_list(const std::vector<EXTLayerRect>& routing_shape_list) { _routing_shape_list = routing_shape_list; }
  void set_cut_shape_list(const std::vector<EXTLayerRect>& cut_shape_list) { _cut_shape_list = cut_shape_list; }
  void set_access_point_list(const std::vector<AccessPoint>& access_point_list) { _access_point_list = access_point_list; }
  // function
  inline std::vector<LayerCoord> getGridCoordList();
  inline std::vector<LayerCoord> getRealCoordList();

 private:
  irt_int _pin_idx = -1;
  std::string _pin_name;
  std::vector<EXTLayerRect> _routing_shape_list;
  std::vector<EXTLayerRect> _cut_shape_list;
  std::vector<AccessPoint> _access_point_list;
};

inline std::vector<LayerCoord> Pin::getGridCoordList()
{
  std::vector<LayerCoord> grid_coord_list;
  for (AccessPoint& access_point : _access_point_list) {
    grid_coord_list.emplace_back(access_point.get_grid_coord(), access_point.get_layer_idx());
  }
  std::sort(grid_coord_list.begin(), grid_coord_list.end(), CmpLayerCoordByXASC());
  grid_coord_list.erase(std::unique(grid_coord_list.begin(), grid_coord_list.end()), grid_coord_list.end());
  return grid_coord_list;
}

inline std::vector<LayerCoord> Pin::getRealCoordList()
{
  std::vector<LayerCoord> real_coord_list;
  for (AccessPoint& access_point : _access_point_list) {
    real_coord_list.emplace_back(access_point.get_real_coord(), access_point.get_layer_idx());
  }
  std::sort(real_coord_list.begin(), real_coord_list.end(), CmpLayerCoordByXASC());
  real_coord_list.erase(std::unique(real_coord_list.begin(), real_coord_list.end()), real_coord_list.end());
  return real_coord_list;
}

}  // namespace irt
