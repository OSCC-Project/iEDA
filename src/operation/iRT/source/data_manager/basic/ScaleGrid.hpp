#pragma once

#include "RTU.hpp"

namespace irt {

class ScaleGrid
{
 public:
  ScaleGrid() = default;
  ~ScaleGrid() = default;
  // getter
  irt_int get_start_line() const { return _start_line; }
  irt_int get_step_length() const { return _step_length; }
  irt_int get_step_num() const { return _step_num; }
  irt_int get_end_line() const { return _end_line; }
  // setter
  void set_start_line(const irt_int start_line) { _start_line = start_line; }
  void set_step_length(const irt_int step_length) { _step_length = step_length; }
  void set_step_num(const irt_int step_num) { _step_num = step_num; }
  void set_end_line(const irt_int end_line) { _end_line = end_line; }
  // function

 private:
  irt_int _start_line = 0;
  irt_int _step_length = 0;
  irt_int _step_num = 0;
  irt_int _end_line = 0;
};

struct CmpScaleGridASC
{
  bool operator()(ScaleGrid& a, ScaleGrid& b) const { return a.get_start_line() < b.get_start_line(); }
};

}  // namespace irt
