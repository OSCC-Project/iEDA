/**
 * @file SAPlacement.hh
 * @author Yuezuo Liu (yuezuoliu@163.com)
 * @brief
 * @version 0.1
 * @date 2023-11-24
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IMP_SAPlacement_H
#define IMP_SAPlacement_H
#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <vector>

#include "Logger.hpp"

namespace imp {

template <typename CoordType, typename RepresentType>
struct SAPlacement
{
  static_assert(std::is_arithmetic<CoordType>::value, "SeqPair requaires a numeric type.");
  template <typename CoordType2, typename RepresentType2>
  friend struct SAAction;

  SAPlacement() = delete;
  SAPlacement(const RepresentType& representation, const std::vector<CoordType>* pin_x_off, const std::vector<CoordType>* pin_y_off,
              const std::vector<CoordType>* lx, const std::vector<CoordType>* ly, const std::vector<CoordType>* dx,
              const std::vector<CoordType>* dy, const std::vector<CoordType>* halo_x, const std::vector<CoordType>* halo_y,
              const std::vector<size_t>* pin2vertex, const std::vector<size_t>* net_span, const std::vector<bool>* is_shape_discrete,
              const std::vector<std::vector<std::pair<CoordType, CoordType>>>* possible_shape_width,
              const std::vector<std::vector<std::pair<CoordType, CoordType>>>* possible_shape_height, CoordType region_lx,
              CoordType region_ly, CoordType region_dx, CoordType region_dy, size_t num_moveable, bool pack_left = true,
              bool pack_bottom = true)

      : _representation(representation),
        _initial_lx(lx),
        _initial_ly(ly),
        _dx(dx),
        _dy(dy),
        _halo_x(halo_x),
        _halo_y(halo_y),
        _pin_x_off(pin_x_off),
        _pin_y_off(pin_y_off),
        _pin2vertex(pin2vertex),
        _net_span(net_span),
        _region_lx(region_lx),
        _region_ly(region_ly),
        _region_dx(region_dx),
        _region_dy(region_dy),
        _num_moveable(num_moveable),
        _pack_left(pack_left),
        _pack_bottom(pack_bottom),
        _is_shape_discrete(is_shape_discrete),
        _possible_shape_width(possible_shape_width),
        _possible_shape_height(possible_shape_height),
        _cost(0)
  {
  }
  SAPlacement(const SAPlacement& other) = default;

  ~SAPlacement() = default;
  SAPlacement& operator=(const SAPlacement& other)
  {
    _representation = other._representation;
    _initial_lx = other._initial_lx;
    _initial_ly = other._initial_ly;
    _dx = other._dx;
    _dy = other._dy;
    _halo_x = other._halo_x;
    _halo_y = other._halo_y;
    _pin_x_off = other._pin_x_off;
    _pin_y_off = other._pin_y_off;
    _pin2vertex = other._pin2vertex;
    _net_span = other._net_span;
    _region_lx = other._region_lx;
    _region_ly = other._region_ly;
    _region_dx = other._region_dx;
    _region_dy = other._region_dy;
    _num_moveable = other._num_moveable;
    _pack_left = other._pack_left;
    _pack_bottom = other._pack_bottom;
    _is_shape_discrete = other._is_shape_discrete;
    _possible_shape_width = other._possible_shape_width;
    _possible_shape_height = other._possible_shape_height;
    _cost = other._cost;
    return *this;
  }

  void randomizeRepresentation()
  {
    _representation.randomize();
    // packing();
  }
  // setter
  void set_representation(const RepresentType& representation) { _representation = representation; }
  void set_pin_x_off(const std::vector<CoordType>* pin_x_off) { _pin_x_off = pin_x_off; }
  void set_pin_y_off(const std::vector<CoordType>* pin_y_off) { _pin_y_off = pin_y_off; }
  // void set_lx(const std::vector<CoordType>& lx) { _lx = lx; }
  // void set_ly(const std::vector<CoordType>& ly) { _ly = ly; }
  void set_dx(const std::vector<CoordType>* dx) { _dx = dx; }
  void set_dy(const std::vector<CoordType>* dy) { _dy = dy; }
  void set_pin2vertex(const std::vector<size_t>* pin2vertex) { _pin2vertex = pin2vertex; }
  void set_net_span(const std::vector<size_t>* net_span) { _net_span = net_span; }
  void set_region_lx(const CoordType& region_lx) { _region_lx = region_lx; }
  void set_region_ly(const CoordType& region_ly) { _region_ly = region_ly; }
  void set_region_dx(const CoordType& region_dx) { _region_dx = region_dx; }
  void set_region_dy(const CoordType& region_dy) { _region_dy = region_dy; }
  void set_num_moveable(const CoordType& num_moveable) { _num_moveable = num_moveable; }
  void set_cost(double cost) { _cost = cost; }

  // getter
  const RepresentType& get_representation() const { return _representation; }
  // RepresentType& get_representation_reference() { return _representation; }
  const std::vector<CoordType>* get_pin_x_off() const { return _pin_x_off; }
  const std::vector<CoordType>* get_pin_y_off() const { return _pin_y_off; }
  const std::vector<CoordType>* get_initial_lx() const { return _initial_lx; }
  const std::vector<CoordType>* get_initial_ly() const { return _initial_ly; }
  const std::vector<CoordType>* get_dx() const { return _dx; }
  const std::vector<CoordType>* get_dy() const { return _dy; }
  const std::vector<size_t>* get_pin2vertex() const { return _pin2vertex; }
  const std::vector<size_t>* get_net_span() const { return _net_span; }
  const CoordType& get_region_lx() const { return _region_lx; }
  const CoordType& get_region_ly() const { return _region_ly; }
  const CoordType& get_region_dx() const { return _region_dx; }
  const CoordType& get_region_dy() const { return _region_dy; }
  const size_t& get_num_moveable() const { return _num_moveable; }
  double get_cost() const { return _cost; }
  bool is_pack_left() const { return _pack_left; }
  bool is_pack_bottom() const { return _pack_bottom; }
  std::pair<CoordType, CoordType> packing(std::vector<CoordType>& lx, std::vector<CoordType>& ly)
  {
    return _representation.packing(*_dx, *_dy, *_halo_x, *_halo_y, lx, ly, _region_lx, _region_ly, _pack_left, _pack_bottom);
  }

 private:
  RepresentType _representation;
  const std::vector<CoordType>* _initial_lx;
  const std::vector<CoordType>* _initial_ly;
  const std::vector<CoordType>* _dx;
  const std::vector<CoordType>* _dy;
  const std::vector<CoordType>* _halo_x;
  const std::vector<CoordType>* _halo_y;
  const std::vector<CoordType>* _pin_x_off;
  const std::vector<CoordType>* _pin_y_off;
  const std::vector<size_t>* _pin2vertex;
  const std::vector<size_t>* _net_span;
  CoordType _region_lx;
  CoordType _region_ly;
  CoordType _region_dx;
  CoordType _region_dy;
  size_t _num_moveable;
  bool _pack_left = true;
  bool _pack_bottom = true;
  const std::vector<bool>* _is_shape_discrete;
  const std::vector<std::vector<std::pair<CoordType, CoordType>>>* _possible_shape_width;   // width_min, width_max
  const std::vector<std::vector<std::pair<CoordType, CoordType>>>* _possible_shape_height;  // height_min, height_max
  double _cost;
};

}  // namespace imp
#include "SAPlacement.tpp"
#endif