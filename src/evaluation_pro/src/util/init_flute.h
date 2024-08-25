

#pragma once

#include "flute.h"

namespace ieval {

class InitFlute
{
 public:
  InitFlute();
  ~InitFlute();

  void readLUT();
  void deleteLUT();
  void printTree(Flute::Tree flutetree);
  void freeTree(Flute::Tree flutetree);
  Flute::Tree flute(int d, int* x, int* y, int acc);
};

}  // namespace ieval