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
#ifndef IDB_MINIMUM_CUT_CHECK_H
#define IDB_MINIMUM_CUT_CHECK_H
#include <vector>

#include "IdbTechEnum.h"

namespace idb {
  class IdbMinimumCutCheck {
   public:
    IdbMinimumCutCheck()
        : _num_cuts(-1),
          _width(-1),
          _cut_distance(-1),
          _connection(MinimumcutConnectionEnum::kDEFAULT),
          _length(-1),
          _distance(-1) { }
    ~IdbMinimumCutCheck() { }
    // getter
    int get_num_cuts() const { return _num_cuts; }
    int get_width() const { return _width; }
    bool has_cut_distance() const { return !(_cut_distance == -1); }
    int get_cut_distance() const { return _cut_distance; }
    bool has_connection() const { return !(_connection == MinimumcutConnectionEnum::kDEFAULT); }
    MinimumcutConnectionEnum get_connection() const { return _connection; }
    bool has_length() const { return !(_length == -1); }
    int get_length() const { return _length; }
    int get_distance() const { return _distance; }
    // setter
    void set_num_cuts(int num_cuts) { _num_cuts = num_cuts; }
    void set_width(int width) { _width = width; }
    void set_cut_distance(int cut_distance) { _cut_distance = cut_distance; }
    void set_connection(MinimumcutConnectionEnum connection) { _connection = connection; }
    void set_length(int length) { _length = length; }
    void set_distance(int distance) { _distance = distance; }
    void setLength(int length, int distance) {
      _length   = length;
      _distance = distance;
    }
    // others
    void setConnection(std::string connectFrom) {
      if (connectFrom == "FROMABOVE") {
        _connection = MinimumcutConnectionEnum::kFROMABOVE;
      } else if (connectFrom == "FROMBELOW") {
        _connection = MinimumcutConnectionEnum::kFROMBELOW;
      }
    }

   private:
    int _num_cuts;
    int _width;
    int _cut_distance;
    MinimumcutConnectionEnum _connection;
    int _length;
    int _distance;  // distance between cut and widewire
  };

  class IdbMinimumCutCheckList {
   public:
    IdbMinimumCutCheckList() { }
    ~IdbMinimumCutCheckList() { }
    void addMinimumCutCheck(std::unique_ptr<IdbMinimumCutCheck> &check) { _minimum_cut_checks.push_back(std::move(check)); }
    IdbMinimumCutCheck *getFirstMinimumCutCheck() { return (_minimum_cut_checks.begin())->get(); }

   private:
    std::vector<std::unique_ptr<IdbMinimumCutCheck>> _minimum_cut_checks;
  };
}  // namespace idb

#endif
