#ifndef SRC_PLATFORM_EVALUATOR_DATA_TIMINGPIN_HPP_
#define SRC_PLATFORM_EVALUATOR_DATA_TIMINGPIN_HPP_

#include <string>

#include "EvalPoint.hpp"

namespace eval {

class TimingPin
{
 public:
  TimingPin() = default;
  ~TimingPin() = default;

  // getter
  std::string& get_name() { return _name; }
  Point<int64_t>& get_coord() { return _coord; }
  int get_layer_id() const { return _layer_id; }
  int get_id() const { return _id; }

  // setter
  void set_name(const std::string& pin_name) { _name = pin_name; }
  void set_coord(const Point<int64_t>& coord) { _coord = coord; }
  void set_layer_id(const int layer_id) { _layer_id = layer_id; }
  void set_id(const int pin_id) { _id = pin_id; }

  // booler
  bool isRealPin() { return _is_real_pin; }
  void set_is_real_pin(const bool is_real_pin) { _is_real_pin = is_real_pin; }

 private:
  std::string _name;
  Point<int64_t> _coord;
  int _layer_id = 1;
  bool _is_real_pin = false;
  int _id = -1;
};

}  // namespace eval

#endif // SRC_PLATFORM_EVALUATOR_DATA_TIMINGPIN_HPP_
