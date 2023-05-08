#ifndef SRC_PLATFORM_EVALUATOR_SOURCE_GDS_WRITER_DATABASE_GDSVIANODES_HPP_
#define SRC_PLATFORM_EVALUATOR_SOURCE_GDS_WRITER_DATABASE_GDSVIANODES_HPP_

#include "EvalPoint.hpp"
#include "EvalRect.hpp"

namespace eval {

class GDSViaNodes
{
 public:
  GDSViaNodes() = default;
  ~GDSViaNodes() = default;

  int32_t get_above_layer_idx() const { return _above_layer_idx; }
  int32_t get_below_layer_idx() const { return _below_layer_idx; }
  Point<int32_t> get_real_coord() const { return _real_coord; }
  Rectangle<int32_t> get_above_shape() const { return _above_shape; }
  Rectangle<int32_t> get_below_shape() const { return _below_shape; }

  void set_above_layer_idx(const int32_t& above_layer_idx) { _above_layer_idx = above_layer_idx; }
  void set_below_layer_idx(const int32_t& below_layer_idx) { _below_layer_idx = below_layer_idx; }
  void set_real_coord(const Point<int32_t>& coord) { _real_coord = coord; }
  void set_above_shape(const Rectangle<int32_t>& shape) { _above_shape = shape; }
  void set_below_shape(const Rectangle<int32_t>& shape) { _below_shape = shape; }

 private:
  int32_t _above_layer_idx = -1;
  int32_t _below_layer_idx = -1;
  Point<int32_t> _real_coord;
  Rectangle<int32_t> _above_shape;
  Rectangle<int32_t> _below_shape;
};

}  // namespace eval

#endif  // SRC_PLATFORM_EVALUATOR_SOURCE_GDS_WRITER_DATABASE_GDSVIANODES_HPP_
