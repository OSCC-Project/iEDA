// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <unordered_map>

#include "MPSolution.hh"
#include "Setting.hh"
#include "module/logger/Log.hh"

namespace ipl::imp {
class SequencePair : public MPSolution
{
 public:
  SequencePair(std::vector<FPInst*> macro_list, Setting* set);
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
  Orient _old_orient = Orient::kN;

  void singleSwap(bool flag);  // true for pos_seq and false for neg_seq
  void doubleSwap(int index1, int index2);
  void pl2sp(std::vector<FPInst*> macro_list);
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