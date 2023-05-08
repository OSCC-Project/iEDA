#ifndef SRC_PLATFORM_EVALUATOR_DATA_CONGPIN_HPP_
#define SRC_PLATFORM_EVALUATOR_DATA_CONGPIN_HPP_

#include <climits>
#include <string>
#include <vector>

#include "EvalPoint.hpp"
#include "EvalType.hpp"

namespace eval {

class CongPin
{
 public:
  CongPin() = default;
  ~CongPin() = default;

  // getter
  std::string get_name() const { return _name; }
  Point<int64_t> get_coord() const { return _coord; }
  int64_t get_x() const { return _coord.get_x(); }
  int64_t get_y() const { return _coord.get_y(); }
  PIN_TYPE get_type() const { return _type; }
  int64_t get_area() const { return _area; }

  // setter
  void set_name(const std::string& pin_name) { _name = pin_name; }
  void set_coord(const Point<int64_t>& coord) { _coord = coord; }
  void set_x(const int64_t& x) { _coord.set_x(x); }
  void set_y(const int64_t& y) { _coord.set_y(y); }
  void set_type(const PIN_TYPE& pin_type) { _type = pin_type; }
  void set_area(const int64_t& pin_area) { _area = pin_area; }

  // booler
  bool isIOPort() const { return _type == PIN_TYPE::kIOPort; }
  bool isInstancePort() const { return _type == PIN_TYPE::kInstancePort; }

 private:
  std::string _name;
  Point<int64_t> _coord;
  PIN_TYPE _type;
  int64_t _area;
};

}  // namespace eval

#endif // SRC_PLATFORM_EVALUATOR_DATA_CONGPIN_HPP_
