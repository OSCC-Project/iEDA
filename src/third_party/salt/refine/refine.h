#pragma once

#include "salt/base/tree.h"

namespace salt {

class Refine
{
 public:
  static void cancelIntersect(Tree& tree);
  static void flip(Tree& tree);
  static void uShift(Tree& tree);  // should be after flip to achieve good quality
  static void removeRedundantCoincident(Tree& tree);
  static void substitute(Tree& tree, double eps, bool useRTree = true);
};

}  // namespace salt