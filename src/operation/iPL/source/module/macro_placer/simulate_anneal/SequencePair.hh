#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>

#include "MPSolution.hh"
#include "Setting.hh"

namespace ipl::imp {
class SequencePair : public MPSolution
{
 public:
  SequencePair(vector<FPInst*> macro_list, Setting* set) : MPSolution(macro_list)
  {
    for (int i = 0; i < _num_macro; ++i) {
      _pos_seq.emplace_back(i);
      _neg_seq.emplace_back(i);
      _pre_pos_seq.emplace_back(i);
      _pre_neg_seq.emplace_back(i);
    }
  }
  void perturb() override;
  void pack() override;
  void rollback() override;
  void update() override;
  void printSolution() override;

 private:
  std::vector<int> _pos_seq;
  std::vector<int> _neg_seq;
  std::vector<int> _pre_pos_seq;
  std::vector<int> _pre_neg_seq;

  float _swap_pos_pro = 0.3;
  float _swap_neg_pro = 0.3;
  float _rotate_pro = 0;

  bool _rotate = false;
  int _rotate_macro_index = 0;
  Orient _old_orient = Orient::N;

  void singleSwap(bool flag);  // true for pos_seq and false for neg_seq
  void doubleSwap(int index1, int index2);
  void pl2sp(vector<FPInst*> macro_list);
};

struct RowElem
{
  unsigned _index;
  float _xloc;
};

struct less_mag
{
  bool operator()(const RowElem& elem1, const RowElem& elem2) { return (elem1._xloc < elem2._xloc); }
};

}  // namespace ipl::imp