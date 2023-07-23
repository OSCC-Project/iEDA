#pragma once

#include "salt/base/tree.h"

namespace salt {

class Refine {
public:
    static void CancelIntersect(Tree& tree);
    static void Flip(Tree& tree);
    static void UShift(Tree& tree);  // should be after Flip to achieve good quality
    static void Substitute(Tree& tree, double eps, bool useRTree = true);
};

}  // namespace salt