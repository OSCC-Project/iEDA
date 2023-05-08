#ifndef IDRC_SRC_DB_DRC_ENCLOSURE_H_
#define IDRC_SRC_DB_DRC_ENCLOSURE_H_
#include "DrcEnum.h"
#include "DrcRectangle.h"
namespace idrc {
class DrcEnclosure
{
 public:
  explicit DrcEnclosure() : _layer_idx(0), _shape(DrcRectangle<int>()) {}
  DrcEnclosure(const DrcEnclosure& other)
  {
    _layer_idx = other._layer_idx;
    _shape = other._shape;
    // _direction = other._direction;
  }
  DrcEnclosure(DrcEnclosure&& other)
  {
    _layer_idx = std::move(other._layer_idx);
    _shape = std::move(other._shape);
    // _direction = std::move(other._direction);
  }
  ~DrcEnclosure() {}
  DrcEnclosure& operator=(const DrcEnclosure& other)
  {
    _layer_idx = other._layer_idx;
    _shape = other._shape;
    // _direction = other._direction;
    return (*this);
  }
  DrcEnclosure& operator=(DrcEnclosure&& other)
  {
    _layer_idx = std::move(other._layer_idx);
    _shape = std::move(other._shape);
    // _direction = std::move(other._direction);
    return (*this);
  }
  // getter
  int get_layer_idx() const { return _layer_idx; }
  DrcRectangle<int>& get_shape() { return _shape; }
  // DrcDirection get_direction() { return _direction; }
  // setters
  void set_shape(DrcRectangle<int>& shape) { _shape = shape; }
  void set_layer_idx(int layer_idx) { _layer_idx = layer_idx; }
  // void set_direction(DrcDirection direction) { _direction = direction; }
  // function

 private:
  int _layer_idx;
  DrcRectangle<int> _shape;
  // DrcDirection _direction;
};
}  // namespace idrc

#endif