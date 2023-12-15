/**
 * @file LibertyClassifyCell.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief classify the liberty cell according the liberty
 * cell、port、arc、function.
 * @version 0.1
 * @date 2023-12-05
 *
 * @copyright Copyright (c) 2023
 *
 */
#pragma once

#include <unordered_map>

#include "BTreeMap.hh"
#include "Liberty.hh"
#include "Vector.hh"

namespace ista {

/**
 * @brief class for classify the lib cell.
 *
 */
class LibertyClassifyCell {
 public:
  void classifyLibCell(std::vector<LibertyLibrary*>& the_libs);
  Vector<LibertyCell*>* getClassOfCell(LibertyCell* cell) {
    if (_func_same_cells.contains(cell)) {
      return &_func_same_cells[cell];
    }
    return nullptr;
  }

 private:
  std::size_t hashCellPort(LibertyPort* port);
  std::size_t hashCellPortFuncExpr(RustLibertyExpr* expr);

  std::size_t calculateCellHash(LibertyCell* the_cell);

  bool comparePort(LibertyPort* port1, LibertyPort* port2);
  bool comparePortFunc(RustLibertyExpr* expr1, RustLibertyExpr* expr2);
  bool compareCellPortsAndFuncs(LibertyCell* cell1, LibertyCell* cell2);

  bool compareCellTimingArc(LibertyArcSet* set1, LibertyArcSet* set2);
  bool compareCellTimingArcSets(LibertyCell* cell1, LibertyCell* cell2);

  bool compareCellFunction(LibertyCell* the_cell1, LibertyCell* the_cell2);

  void classifyOneLibCell(
      LibertyLibrary* the_lib,
      std::unordered_map<std::size_t, Vector<LibertyCell*>>& hash_to_cells);

  ieda::BTreeMap<LibertyCell*, Vector<LibertyCell*>>
      _func_same_cells;  //!< The one cell map to the func same cell with
                         //!< different size.
};
}  // namespace ista