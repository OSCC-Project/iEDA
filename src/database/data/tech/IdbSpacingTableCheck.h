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
#ifndef IDB_SPACING_TABLE_CHECK
#define IDB_SPACING_TABLE_CHECK
#include <map>
#include <memory>
#include <vector>

#include "IdbLookupTable.h"

namespace idb {
  class IdbSpacingTableParallelRunLengthCheck {
   public:
    IdbSpacingTableParallelRunLengthCheck() { }
    IdbSpacingTableParallelRunLengthCheck(std::unique_ptr<Idb2DLookupTable<int, int, int>> checkTable)
        : _parallel_run_length_check(std::move(checkTable)) { }
    virtual ~IdbSpacingTableParallelRunLengthCheck() { }
    // getter
    std::unique_ptr<Idb2DLookupTable<int, int, int>> get_parallel_run_length_check() {
      return std::move(_parallel_run_length_check);
    }
    // setter
    void set_parallel_run_length_check(std::unique_ptr<Idb2DLookupTable<int, int, int>> checkTable) {
      _parallel_run_length_check = std::move(checkTable);
    }
    // others
    int findSpacing(int width, int parallelRunLength) {
      return _parallel_run_length_check->findValue(width, parallelRunLength);
    }

   protected:
    std::unique_ptr<Idb2DLookupTable<int, int, int>> _parallel_run_length_check;
  };
  /******************************************************************************************/
  class IdbSpacingTableTwoWidthIndex {
   public:
    IdbSpacingTableTwoWidthIndex() : _width(0), _prl(0), _has_prl(false) { }
    IdbSpacingTableTwoWidthIndex(int width, int prl = 0, bool hasPrl = false)
        : _width(width), _prl(prl), _has_prl(hasPrl) { }
    ~IdbSpacingTableTwoWidthIndex() { }
    bool operator<(const IdbSpacingTableTwoWidthIndex &index) const {
      return _has_prl ? (_width < index._width && _prl < index._prl) : (_width < index._width);
    }

   private:
    int _width;
    int _prl;
    bool _has_prl;
  };
  class IdbSpacingTableTwoWidthCheck {
   public:
    IdbSpacingTableTwoWidthCheck() { }
    IdbSpacingTableTwoWidthCheck(
        const Idb2DLookupTable<IdbSpacingTableTwoWidthIndex, IdbSpacingTableTwoWidthIndex, int> &table)
        : _two_width_spacing_table(table) { }
    ~IdbSpacingTableTwoWidthCheck() { }
    // getter
    const Idb2DLookupTable<IdbSpacingTableTwoWidthIndex, IdbSpacingTableTwoWidthIndex, int> &get_two_width_spacing_table()
        const {
      return _two_width_spacing_table;
    }
    // setter
    void set_two_width_spacing_table(
        const Idb2DLookupTable<IdbSpacingTableTwoWidthIndex, IdbSpacingTableTwoWidthIndex, int> &table) {
      _two_width_spacing_table = table;
    }
    // others
    int find(const int &firstWidth, const int &secondWidth, int prl) const {
      return _two_width_spacing_table.findValue(IdbSpacingTableTwoWidthIndex(firstWidth, prl),
                                                IdbSpacingTableTwoWidthIndex(secondWidth, prl));
    }
    int findMin() const { return _two_width_spacing_table.findMin(); }

   private:
    Idb2DLookupTable<IdbSpacingTableTwoWidthIndex, IdbSpacingTableTwoWidthIndex, int> _two_width_spacing_table;
  };
  /*********************************************************************************************************/
  class IdbLef58SpacingTableCheck : public IdbSpacingTableParallelRunLengthCheck {
   public:
    IdbLef58SpacingTableCheck() { }
    IdbLef58SpacingTableCheck(std::unique_ptr<Idb2DLookupTable<int, int, int>> parallelRunLengthCheck,
                              const std::map<int, std::pair<int, int>> &exceptWithinCheck)
        : IdbSpacingTableParallelRunLengthCheck(std::move(parallelRunLengthCheck)),
          _except_within_check(exceptWithinCheck),
          _wrong_direction(false),
          _same_mask(false),
          _except_eol(false),
          _eol_width(0) { }
    ~IdbLef58SpacingTableCheck() { }
    // getter
    bool get_wrong_direction() const { return _wrong_direction; }
    bool get_same_mask() const { return _same_mask; }
    bool get_except_eol() const { return _except_eol; }
    int get_eol_width() const { return _eol_width; }
    // setter
    void set_except_within_check(std::map<int, std::pair<int, int>> except_within_check) {
      _except_within_check = except_within_check;
    }
    void set_wrong_direction(bool wrong_direction) { _wrong_direction = wrong_direction; }
    void set_same_mask(bool same_mask) { _same_mask = same_mask; }
    void set_eol_width(int eol_width) {
      _except_eol = true;
      _eol_width  = eol_width;
    }
    // others
    bool hasExceptWithin(int rowId) const {
      auto rowPosition = _parallel_run_length_check->getRowPosition(rowId);
      return _except_within_check.find(rowPosition) != _except_within_check.end();
    }
    std::pair<int, int> getExceptWithin(int rowId) {
      auto rowPosition = _parallel_run_length_check->getRowPosition(rowId);
      return _except_within_check.at(rowPosition);
    }

   private:
    std::map<int, std::pair<int, int>> _except_within_check;
    bool _wrong_direction;
    bool _same_mask;
    bool _except_eol;
    int _eol_width;  // unsigned?
  };
}  // namespace idb

#endif
