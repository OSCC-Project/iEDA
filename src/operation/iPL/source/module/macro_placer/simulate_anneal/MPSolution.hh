#pragma once
#include <string>
#include <vector>

#include "Solution.hh"
#include "database/FPInst.hh"

namespace ipl::imp {
class MPSolution : public Solution
{
 public:
  MPSolution(vector<FPInst*> macro_list)
  {
    _num_macro = macro_list.size();
    _macro_list = macro_list;
  }
  uint32_t get_total_width() { return _total_width; }
  uint32_t get_total_height() { return _total_height; }
  float get_total_area() { return _total_area; }
  virtual void printSolution(){};

 protected:
  int _num_macro = 0;
  vector<FPInst*> _macro_list;
  uint32_t _total_width = 0;
  uint32_t _total_height = 0;
  float _total_area = 0;
};

}  // namespace ipl::imp