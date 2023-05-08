#pragma once
#include <stdint.h>

namespace ipl {

class Solution
{
 public:
  Solution(){};
  virtual void perturb() = 0;
  virtual void pack() = 0;
  virtual void rollback() = 0;
  virtual void update() = 0;
};

}  // namespace ipl