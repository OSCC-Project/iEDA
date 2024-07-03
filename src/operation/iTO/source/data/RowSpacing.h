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
#include <cassert>
#include <iomanip>
#include <iostream>
#include <set>
#include <vector>
#include <array>

namespace ito {
class UsedSpace;
using UsedSpace = UsedSpace;
using std::array;
using std::cout;
using std::endl;
using std::ios;
using std::ofstream;
using std::pair;
using std::set;
using std::setw;
using std::string;
using std::vector;

class UsedSpace {
 public:
  UsedSpace() = default;
  UsedSpace(int begin, int end) : _begin(begin), _end(end) {
    assert(end >= begin);
    _length = end - begin;
  }
  ~UsedSpace() = default;

  int begin() { return _begin; }
  int end() { return _end; }

  int length() { return _length; }

  bool operator==(UsedSpace *used) {
    return this->_begin == used->begin() && this->_end == used->end();
  }

  /**
   * @brief space "used" inside the space -> true
   */
  bool isOverlaps(UsedSpace *used) {
    return this->_begin <= used->begin() && this->_end >= used->end();
  }

  /**
   * @brief space "used" inside the space -> true
   */
  bool isOverlaps(int begin, int end) {
    return this->_begin <= begin && this->_end >= end;
  }

  bool antiOverlaps(UsedSpace *used) { return used->isOverlaps(this); }

  bool antiOverlaps(int begin, int end) {
    return this->_begin >= begin && this->_end <= end;
  }

  /**
   * @brief space "used" not inside the space, but there is an intersection between them
   */
  bool isCross(UsedSpace *used) {
    if (this->isOverlaps(used)) {
      return false;
    }
    return (used->end() > this->_begin && used->begin() < this->_begin) ||
           (used->begin() < this->_end && used->end() > this->_end);
    // return isOverlaps(used->begin(), used->end());
  }

  /**
   * @brief space "used" not inside the space, but there is an intersection between them
   */
  bool isCross(int begin, int end) {
    if (this->isOverlaps(begin, end)) {
      return false;
    }
    return (end > this->_begin && begin < this->_begin) ||
           (begin < this->_end && end > this->_end);
    //  return (end > this->_begin && end < this->_end) ||
    //  (begin > this->_begin && begin < this->_end);
  }

 private:
  int _begin = 0;
  int _end = 0;
  int _length = 0;
};

class CompareSpace {
 public:
  bool operator()(UsedSpace *u1, UsedSpace *u2) { return u1->begin() < u2->begin(); }
};

class RowSpacing {
 public:
  RowSpacing() = default;
  RowSpacing(int begin, int end) : _begin(begin), _end(end) {
    UsedSpace *unused = new UsedSpace(begin, end);
    _unused_space.push_back(unused);
  }
  ~RowSpacing() = default;

  inline void addUsedSpace(UsedSpace *used);
  inline void addUsedSpace(int begin, int end);

  inline vector<UsedSpace *> searchFeasiblePlace(int begin, int end, int find_num);

  void printUsed() {
    for (auto u : _used_space) {
      cout << u->begin() << "  " << u->end() << endl;
    }
  }
  void printUnused() {
    cout << "unused " << endl;
    for (auto u : _unused_space) {
      cout << u->begin() << "  " << u->end() << endl;
    }
  }

 private:
  inline bool isLegalSpace(int expect_space_length, UsedSpace *space,
                           vector<UsedSpace *> &unused);

  int                 _begin = 0;
  int                 _end = 0;
  vector<UsedSpace *> _used_space = {};
  vector<UsedSpace *> _unused_space = {};
};

inline void RowSpacing::addUsedSpace(UsedSpace *used) {
  _used_space.push_back(used);
  // change the unused space
  // case 1:
  // for (auto unuse : _unused_space) {
  for (int i = 0; i < (int)_unused_space.size(); i++) {
    auto unuse = _unused_space[i];
    if (unuse->isOverlaps(used)) {
      // 1. remove
      auto iter = std::remove(_unused_space.begin(), _unused_space.end(), unuse);
      _unused_space.erase(iter, _unused_space.end());

      if (unuse == used) {
        break;
      }

      // 2. update
      if (unuse->begin() < used->begin()) {
        UsedSpace *unused1 = new UsedSpace(unuse->begin(), used->begin());
        // _unused_space.push_back(unused1);
        _unused_space.insert(std::upper_bound(_unused_space.begin(), _unused_space.end(),
                                              unused1, CompareSpace()),
                             unused1);
      }
      if (used->end() < unuse->end()) {
        UsedSpace *unused2 = new UsedSpace(used->end(), unuse->end());
        // _unused_space.push_back(unused2);
        _unused_space.insert(std::upper_bound(_unused_space.begin(), _unused_space.end(),
                                              unused2, CompareSpace()),
                             unused2);
      }
      break;
    }
    // case 2:  "usedSpace" occupy multiple "unusedSpaces"
    else if (unuse->isCross(used)) {
      auto last_unuse = unuse;
      int  occupy_num = 0;
      bool end_point_in_used_space = false;
      for (int j = 0; j < (int)_unused_space.size() - i; j++) {
        if (used->end() < _unused_space[i + j]->end()) {
          last_unuse = _unused_space[i + j];
          if (used->end() < _unused_space[i + j]->begin()) {
            end_point_in_used_space = true;
            occupy_num = j - 1;
            break;
          }
          occupy_num = j;
          break;
        }
      }

      // 1. remove
      for (int earse = 0; earse < occupy_num + 1; earse++) {
        auto iter =
            std::remove(_unused_space.begin(), _unused_space.end(), _unused_space[i]);
        _unused_space.erase(iter, _unused_space.end());
      }

      // 2. update
      if (unuse->begin() < used->begin()) {
        UsedSpace *unused1 = new UsedSpace(unuse->begin(), used->begin());
        // _unused_space.push_back(unused1);
        _unused_space.insert(std::upper_bound(_unused_space.begin(), _unused_space.end(),
                                              unused1, CompareSpace()),
                             unused1);
      }
      if (used->end() < last_unuse->end() && !end_point_in_used_space) {
        UsedSpace *unused2 = new UsedSpace(used->end(), last_unuse->end());
        // _unused_space.push_back(unused2);
        _unused_space.insert(std::upper_bound(_unused_space.begin(), _unused_space.end(),
                                              unused2, CompareSpace()),
                             unused2);
      }
      break;
    }
  }
}

inline void RowSpacing::addUsedSpace(int begin, int end) {
  UsedSpace *used = new UsedSpace(begin, end);
  _used_space.push_back(used);
  addUsedSpace(used);
}

/**
 * @brief find nearly feasible place
 *
 * @param begin
 * @param end
 * @param find_num The number you want to search
 * @return vector<UsedSpace *>
 */
inline vector<UsedSpace *> RowSpacing::searchFeasiblePlace(int begin, int end,
                                                           int find_num) {
  assert(end >= begin);
  sort(_unused_space.begin(), _unused_space.end(),
       [](UsedSpace *u1, UsedSpace *u2) { return u1->begin() < u2->begin(); });

  int  unused_num = _unused_space.size();
  int  search_end = _unused_space.size() - 1;
  int  search_begin = 0;
  int  search_mid = 0;
  bool excute = true;

  vector<UsedSpace *> unused;

  int expect_space_length = abs(end - begin);
  if (search_end <= 3) {
    for (auto u : _unused_space) {
      if (u->length() >= expect_space_length) {
        unused.push_back(u);
      }
    }
    return unused;
  }

  int find_count = 0;
  while (excute) {
    search_mid = (search_begin + search_end) / 2;
    if (_unused_space[search_mid]->begin() <= begin &&
        _unused_space[search_mid + 1]->begin() >= begin) {
      excute = false;
      if (_unused_space[search_mid]->isOverlaps(begin, end)) {
        unused.push_back(_unused_space[search_mid]);
        // case
        return unused;
      }
      break;
    }

    if (_unused_space[search_mid]->begin() > begin) {
      search_end = search_mid;
    } else {
      search_begin = search_mid;
    }

    if (search_begin >= search_end || search_end - search_begin == 1) {
      break;
    }
  }

  int find = 1;
  while (find_num > find_count) {
    if (search_mid + find < unused_num) {
      if (isLegalSpace(expect_space_length, _unused_space[search_mid + find], unused)) {
        find_count++;
      }
    }
    if (search_mid - find >= 0) {
      if (isLegalSpace(expect_space_length, _unused_space[search_mid - find], unused)) {
        find_count++;
      }
    }
    if (search_mid - find < 0 && search_mid + find >= unused_num) {
      break;
    }
    find++;
  }
  return unused;
}

inline bool RowSpacing::isLegalSpace(int expect_space_length, UsedSpace *space,
                                     vector<UsedSpace *> &unused) {
  if (space->length() >= expect_space_length) {
    unused.push_back(space);
    return true;
  }
  return false;
}
} // namespace ito