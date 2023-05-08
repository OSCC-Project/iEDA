#pragma once

#include "PlanarCoord.hpp"
#include "RTU.hpp"

namespace irt {

class GPPath : public Segment<PlanarCoord>
{
 public:
  GPPath(const Segment<PlanarCoord>& segment, const irt_int layer_idx, const irt_int data_type, const irt_int width)
      : Segment<PlanarCoord>(segment)
  {
    _layer_idx = layer_idx;
    _data_type = data_type;
    _width = width;
  }
  GPPath() = default;
  ~GPPath() = default;
  // getter
  irt_int get_layer_idx() const { return _layer_idx; }
  irt_int get_data_type() const { return _data_type; }
  irt_int get_width() const { return _width; }
  Segment<PlanarCoord>& get_segment() { return (*this); }
  // setter
  void set_layer_idx(const irt_int layer_idx) { _layer_idx = layer_idx; }
  void set_data_type(const irt_int data_type) { _data_type = data_type; }
  void set_width(const irt_int width) { _width = width; }
  void set_segment(const Segment<PlanarCoord>& segment)
  {
    set_first(segment.get_first());
    set_second(segment.get_second());
  }
  void set_segment(const PlanarCoord& first_coord, const PlanarCoord& second_coord)
  {
    set_first(first_coord);
    set_second(second_coord);
  }
  void set_segment(const irt_int first_x, const irt_int first_y, const irt_int second_x, const irt_int second_y)
  {
    set_first(PlanarCoord(first_x, first_y));
    set_second(PlanarCoord(second_x, second_y));
  }
  // function

 private:
  irt_int _layer_idx = -1;
  irt_int _data_type = 0;
  irt_int _width = 1;
};

}  // namespace irt
