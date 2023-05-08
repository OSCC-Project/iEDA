#ifndef SRC_EVALUATOR_SOURCE_CONGESTION_DATABASE_CONGINST_HPP_
#define SRC_EVALUATOR_SOURCE_CONGESTION_DATABASE_CONGINST_HPP_

#include "CongPin.hpp"
#include "EvalRect.hpp"

namespace eval {

class CongInst
{
 public:
  CongInst() = default;
  ~CongInst() = default;

  // getter
  std::string get_name() const { return _name; }
  Rectangle<int64_t> get_shape() const { return _shape; }
  int64_t get_width() { return _shape.get_width(); }
  int64_t get_height() { return _shape.get_height(); }
  int64_t get_lx() const { return _shape.get_ll_x(); }
  int64_t get_ly() const { return _shape.get_ll_y(); }
  int64_t get_ux() const { return _shape.get_ur_x(); }
  int64_t get_uy() const { return _shape.get_ur_y(); }
  std::vector<CongPin*> get_pin_list() const { return _pin_list; }
  INSTANCE_TYPE get_type() const { return _type; }

  // booler
  bool isNormalInst() const { return _type == INSTANCE_TYPE::kNormal; }
  bool isOutsideInst() const { return _type == INSTANCE_TYPE::kOutside; }

  // setter
  void set_name(const std::string& inst_name) { _name = inst_name; }
  void set_shape(const int64_t& lx, const int64_t& ly, const int64_t& ux, const int64_t& uy) { _shape.set_rectangle(lx, ly, ux, uy); }
  void set_pin_list(const std::vector<CongPin*>& cong_pin_list) { _pin_list = cong_pin_list; }
  void set_type(const INSTANCE_TYPE& type) { _type = type; }
  void add_pin(CongPin* pin) { _pin_list.push_back(pin); }

 private:
  std::string _name;
  Rectangle<int64_t> _shape;
  std::vector<CongPin*> _pin_list;
  INSTANCE_TYPE _type;
};

}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_CONGESTION_DATABASE_CONGINST_HPP_
