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
#include "SequencePair.hh"

#include <unordered_map>

using namespace std;

namespace ipl::imp {

void SequencePair::perturb()
{
  float rand_num = rand() / (RAND_MAX + 1.0);
  if (rand_num < 1 - _rotate_pro) {
    _rotate = false;
    int block_a = int(_num_macro * (rand() / (RAND_MAX + 1.0)));
    int block_b = int((_num_macro - 1) * (rand() / (RAND_MAX + 1.0)));
    block_b = (block_b >= block_a) ? block_b + 1 : block_b;
    if (rand_num < _swap_pos_pro) {
      swap(_pos_seq[block_a], _pos_seq[block_b]);
    } else if (rand_num < _swap_pos_pro + _swap_neg_pro) {
      swap(_neg_seq[block_a], _neg_seq[block_b]);
    } else {
      doubleSwap(block_a, block_b);
    }
  } else {
    _rotate = true;
    _rotate_macro_index = int(_num_macro * (rand() / (RAND_MAX + 1.0)));
    _old_orient = _macro_list[_rotate_macro_index]->get_orient();
    int new_orient = (_rotate_macro_index + 1) % 8;  // the total of orient is 8
    _macro_list[_rotate_macro_index]->set_orient(Orient(new_orient));
  }
  pack();
}

void SequencePair::pack()
{
  for (int i = 0; i < _num_macro; ++i) {
    _macro_list[i]->set_x(0);
    _macro_list[i]->set_y(0);
  }

  // calculate x position
  vector<pair<int, int>> match(_num_macro);
  for (size_t i = 0; i < _pos_seq.size(); ++i) {
    match[_pos_seq[i]].first = i;
    match[_neg_seq[i]].second = i;
  }

  vector<uint32_t> length(_num_macro);
  for (int i = 0; i < _num_macro; ++i) {
    length[i] = 0;
  }

  for (size_t i = 0; i < _pos_seq.size(); ++i) {
    int b = _pos_seq[i];
    int p = match[b].second;
    _macro_list[b]->set_x(length[p]);
    uint t = _macro_list[b]->get_x() + _macro_list[b]->get_width();
    for (size_t j = p; j < _neg_seq.size(); ++j) {
      if (t > length[j]) {
        length[j] = t;
      } else {
        break;
      }
    }
  }

  _total_width = length[_num_macro - 1];

  // calculate Y position
  vector<int> pos_seq(_pos_seq.size());
  for (int i = 0; i < _num_macro; ++i) {
    pos_seq[i] = _pos_seq[_num_macro - 1 - i];
  }
  for (int i = 0; i < _num_macro; ++i) {
    match[pos_seq[i]].first = i;
    match[_neg_seq[i]].second = i;
  }

  for (int i = 0; i < _num_macro; ++i) {
    length[i] = 0;
  }

  for (int i = 0; i < _num_macro; ++i) {
    int b = pos_seq[i];
    int p = match[b].second;
    _macro_list[b]->set_y(length[p]);
    uint32_t t = _macro_list[b]->get_y() + _macro_list[b]->get_height();
    for (int j = p; j < _num_macro; ++j) {
      if (t > length[j]) {
        length[j] = t;
      } else {
        break;
      }
    }
  }

  _total_height = length[_num_macro - 1];
  _total_area = float(_total_width) * float(_total_height);
}

void SequencePair::rollback()
{
  if (!_rotate) {
    for (int i = 0; i < _num_macro; ++i) {
      _pos_seq[i] = _pre_pos_seq[i];
      _neg_seq[i] = _pre_neg_seq[i];
    }
  } else {
    _macro_list[_rotate_macro_index]->set_orient(_old_orient);
  }
}

void SequencePair::update()
{
  if (!_rotate) {
    for (int i = 0; i < _num_macro; ++i) {
      _pre_pos_seq[i] = _pos_seq[i];
      _pre_neg_seq[i] = _neg_seq[i];
    }
  }
}

void SequencePair::doubleSwap(int index1, int index2)
{
  swap(_pos_seq[index1], _pos_seq[index2]);
  size_t neg_index1 = 0;
  size_t neg_index2 = 0;
  for (int i = 0; i < _num_macro; ++i) {
    if (_pos_seq[index1] == _neg_seq[i]) {
      neg_index1 = i;
    }
    if (_pos_seq[index2] == _neg_seq[i]) {
      neg_index2 = i;
    }
  }
  swap(_neg_seq[neg_index1], _neg_seq[neg_index2]);
}

void SequencePair::printSolution()
{
  for (int i = 0; i < _num_macro; ++i) {
    LOG_INFO << _pos_seq[i] << " ";
  }
  for (int i = 0; i < _num_macro; ++i) {
    LOG_INFO << _neg_seq[i] << " ";
  }
  LOG_INFO << "width : ";
  for (FPInst* macro : _macro_list) {
    LOG_INFO << macro->get_width() << " ";
  }
  LOG_INFO << "height: ";
  for (FPInst* macro : _macro_list) {
    LOG_INFO << macro->get_height() << " ";
  }
  for (FPInst* macro : _macro_list) {
    LOG_INFO << "(" << macro->get_x() << ", " << macro->get_y() << ")"
              << " ";
  }
}

void SequencePair::pl2sp(vector<FPInst*> macro_list)
{
  _pos_seq.clear();
  _neg_seq.clear();
  _pre_pos_seq.clear();
  _pre_neg_seq.clear();
  unsigned size = macro_list.size();
  float row_height = std::numeric_limits<float>::max();
  float max_y_loc = -std::numeric_limits<float>::max();

  for (unsigned i = 0; i < size; ++i) {
    if (row_height > macro_list[i]->get_height())
      row_height = macro_list[i]->get_height();

    if (max_y_loc < macro_list[i]->get_y())
      max_y_loc = macro_list[i]->get_y();
  }

  unsigned num_row_list = unsigned(ceil(max_y_loc / row_height) + 1);

  // snap to y grid here
  for (unsigned i = 0; i < size; ++i) {
    unsigned reqd_row = static_cast<unsigned>((macro_list[i]->get_y() / row_height) + 0.5);
    macro_list[i]->set_y(reqd_row * row_height);
  }

  vector<vector<RowElem>> row_list;
  vector<RowElem> single_row_list;
  RowElem temp_row_elem;

  float curr_height = 0;

  for (unsigned i = 0; i < num_row_list; ++i) {
    for (unsigned j = 0; j < size; ++j) {
      if (std::abs(macro_list[j]->get_y() - curr_height) < 0.0001) {
        temp_row_elem._index = j;
        temp_row_elem._xloc = macro_list[j]->get_x();
        single_row_list.push_back(temp_row_elem);
      }
    }

    curr_height += row_height;

    std::stable_sort(single_row_list.begin(), single_row_list.end(), less_mag());
    row_list.push_back(single_row_list);
    single_row_list.clear();
  }

  // form the X and Y sequence pairs now
  for (unsigned i = 0; i < row_list.size(); ++i) {
    for (unsigned j = 0; j < row_list[i].size(); ++j) {
      _neg_seq.push_back(row_list[i][j]._index);
    }
  }
  for (int i = row_list.size() - 1; i >= 0; --i) {
    for (unsigned j = 0; j < row_list[i].size(); ++j) {
      _pos_seq.push_back(row_list[i][j]._index);
    }
  }

  if (_pos_seq.size() != macro_list.size() || _neg_seq.size() != macro_list.size()) {
    LOG_ERROR << "generated sequence pair of not correct sizes " << _pos_seq.size() << " & " << _neg_seq.size() << " vs " << size << endl;
  }

  pack();
}

}  // namespace ipl::imp