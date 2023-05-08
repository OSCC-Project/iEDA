#pragma once
#include <vector>

#include "BStarTree.hh"
#include "SequencePair.hh"
#include "Setting.hh"
#include "database/FPInst.hh"

namespace ipl::imp {

class SolutionFactory
{
 public:
  MPSolution* createSolution(vector<FPInst*> macro_list, Setting* set)
  {
    switch (set->get_solution_type()) {
      case SolutionTYPE::BST:
        return new BStarTree(macro_list, set);
        break;
      case SolutionTYPE::SP:
        return new SequencePair(macro_list, set);
        break;
      default:
        break;
    }
    return nullptr;
  }
};
}  // namespace ipl::imp