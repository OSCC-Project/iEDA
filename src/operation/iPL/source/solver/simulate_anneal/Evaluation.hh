#pragma once
#include "SAParam.hh"
#include "Solution.hh"

namespace ipl {

class Evaluation
{
 public:
  Evaluation(){};
  virtual float evaluate() = 0;
  virtual void init_norm(SAParam* param) = 0;
  virtual Solution* get_solution() = 0;
  virtual void showMassage() = 0;
};
}  // namespace ipl